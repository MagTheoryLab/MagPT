import argparse
import numpy as np
from ase.io import read
from deepmd.infer import DeepPot

p = argparse.ArgumentParser()
p.add_argument("model")
p.add_argument("xyz")
p.add_argument("-i", "--index", type=int, default=0)
args = p.parse_args()

model = DeepPot(args.model)
sym2idx = {s: i for i, s in enumerate(model.get_type_map())}

atoms = read(args.xyz, index=args.index)
atype = np.array([sym2idx[s] for s in atoms.get_chemical_symbols()])

e, f, v, fm, _ = model.eval(
    coords=atoms.get_positions()[None],
    cells=atoms.cell.array[None],
    spin=atoms.arrays["spin"][None],
    atom_types=atype,
)

print("energy       (eV) :", np.asarray(e).ravel()[0])
print("force      (eV/A) :\n", np.asarray(f).reshape(-1, 3))
print("force_mag (eV/uB) :\n", np.asarray(fm).reshape(-1, 3))
print("virial       (eV) :\n", np.asarray(v).reshape(3, 3))