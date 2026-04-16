from ase.io import read, write
from ase.optimize import BFGS
from ase.calculators.lj import LennardJones

import sys

cif_in = sys.argv[1]
cif_out = sys.argv[2]

atoms = read(cif_in)
atoms.calc = LennardJones()
opt = BFGS(atoms, logfile=None)
opt.run(fmax=0.5, steps=200)
write(cif_out, atoms)
print(f"Relajado: {cif_out}")
