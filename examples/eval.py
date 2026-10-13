import argparse
import numpy as np
from collections import defaultdict
from ase.io import read
from deepmd.infer import DeepPot


def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate a DeePMD model against an extxyz dataset."
    )
    parser.add_argument("model", help="path to the DeePMD model file, e.g. MagPT_399168.pth")
    parser.add_argument("xyz",   help="path to the extxyz file containing one or more systems")
    return parser.parse_args()


def group_by_system(frames):
    """Group frames by element composition (species + order)."""
    groups = defaultdict(list)
    for a in frames:
        groups[tuple(a.get_chemical_symbols())].append(a)
    return groups


def build_inputs(frames, sym2idx):
    """Assemble batched arrays and DFT references for one system."""
    symbols = frames[0].get_chemical_symbols()
    natoms = len(symbols)

    coord = np.stack([a.get_positions() for a in frames])   # (F, N, 3)
    box   = np.stack([a.cell.array     for a in frames])    # (F, 3, 3)
    spin  = np.stack([a.arrays["spin"] for a in frames])    # (F, N, 3)
    atype = np.array([sym2idx[s] for s in symbols])         # (N,)

    ref = {
        "energy":    np.array([a.get_potential_energy() for a in frames]),
        "force":     np.stack([a.arrays["force"]     for a in frames]),
        "force_mag": np.stack([a.arrays["force_mag"] for a in frames]),
        "virial":    np.stack([a.info["virial"]      for a in frames]).reshape(len(frames), -1),
    }
    return coord, box, spin, atype, ref, natoms


def evaluate_system(frames, model, sym2idx):
    """Run one batched inference and return per-atom-normalized residuals (eV)."""
    coord, box, spin, atype, ref, natoms = build_inputs(frames, sym2idx)

    e_model, fr_model, v_model, fm_model, _ = model.eval(
        coords=coord, cells=box, spin=spin, atom_types=atype
    )

    res = {
        "energy":    (ref["energy"] - e_model) / natoms,
        "force":     ref["force"]     - fr_model,
        "force_mag": ref["force_mag"] - fm_model,
        "virial":    (ref["virial"]   - v_model) / natoms,
    }
    return {"natoms": natoms, "nframes": len(frames), "res": res}


def rmse_meV(residuals):
    """RMSE in meV, pooled over all supplied residuals."""
    d = np.concatenate([np.asarray(r, dtype=float).ravel() for r in residuals])
    return float(np.sqrt(np.mean(d ** 2)) * 1000.0)


def main():
    args = parse_args()

    model = DeepPot(args.model)
    type_map = model.get_type_map()
    sym2idx = {s: i for i, s in enumerate(type_map)}
    print("type_map =", type_map)

    frames = read(args.xyz, index=":")
    systems = group_by_system(frames)
    print(f"Loaded {len(frames)} frames in {len(systems)} system(s).\n")

    per_system = []
    pooled = {k: [] for k in ("energy", "force", "force_mag", "virial")}

    for symbols, group in systems.items():
        r = evaluate_system(group, model, sym2idx)
        per_system.append((symbols, r))
        for k, v in r["res"].items():
            pooled[k].append(np.asarray(v, dtype=float).ravel())

    # ---- Per-system results ----
    for symbols, r in per_system:
        label = f"{''.join(symbols)} (N={r['natoms']}, F={r['nframes']})"
        print(f"[{label}]")
        print(f"  Energy RMSE        : {rmse_meV([r['res']['energy']]):>12.6f} meV/atom")
        print(f"  Force RMSE         : {rmse_meV([r['res']['force']]):>12.6f} meV/Å")
        print(f"  Magnetic Force RMSE: {rmse_meV([r['res']['force_mag']]):>12.6f} meV/μB")
        print(f"  Virial RMSE        : {rmse_meV([r['res']['virial']]):>12.6f} meV/atom")
        print()

    # ---- Overall (pooled) results ----
    print(f"[Overall]  {len(systems)} systems, {len(frames)} frames")
    print(f"  Energy RMSE        : {rmse_meV(pooled['energy']):>12.6f} meV/atom")
    print(f"  Force RMSE         : {rmse_meV(pooled['force']):>12.6f} meV/Å")
    print(f"  Magnetic Force RMSE: {rmse_meV(pooled['force_mag']):>12.6f} meV/μB")
    print(f"  Virial RMSE        : {rmse_meV(pooled['virial']):>12.6f} meV/atom")


if __name__ == "__main__":
    main()