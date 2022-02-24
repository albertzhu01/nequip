import torch
import numpy as np
from sklearn import mixture
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from ase.visualize import view
from ase.io import read, write
from ase import Atoms
from sklearn.metrics import mean_absolute_error

from nequip.utils import Config
from nequip.data import AtomicDataDict, AtomicData, Collater
from nequip.nn import SequentialGraphNetwork, SaveForOutput
# matplotlib.rc('xtick', labelsize=20)
# matplotlib.rc('ytick', labelsize=20)
# test = np.random.rand(25)
# print(test)
# ind_smallest10 = np.argpartition(test, 10)[:10]
# ind_largest10 = np.argpartition(test, -10)[-10:]
# print(test[ind_smallest10])
# print(test[ind_largest10])
#
# arr = np.array([1, 3, 2, 4, 5])
# idx = arr.argsort()[-3:][::-1]
# print(idx)
# print(arr[idx])

atoms_list = read("C:/Users/alber/nequip/configs_final/train_300K.xyz", index=":", format="extxyz")
# for idx, atoms in enumerate(atoms_list[:2]):
#     print(atoms)
#     print(atoms.get_forces())
#     for key in atoms:
#         print(key)
# print(atoms)

# view(atoms_list[0])

dihedrals = []
for idx, atoms in enumerate(atoms_list):
    angle = atoms.get_dihedral(1, 5, 13, 14)
    dihedrals.append(angle)
    print(f"molecule {idx}: {angle}")

dihedrals = np.array(dihedrals)
print(f"min dihedral: {np.amin(dihedrals)}")
print(f"max dihedral: {np.amax(dihedrals)}")
print(f"avg dihedral: {np.mean(dihedrals)}")
print(f"top 5 max dihedrals idx: {dihedrals.argsort()[-5:]}")
print(f"top 5 max dihedrals angle: {dihedrals[dihedrals.argsort()[-5:]]}")
print(f"bottom 5 min dihedrals idx: {dihedrals.argsort()[:5]}")
print(f"bottom 5 min dihedrals angle: {dihedrals[dihedrals.argsort()[:5]]}")

#
# plt.hist(dihedrals, bins=36, range=(0, 360))
# plt.title("BPA Bridge Dihedral Angle Distribution at 1200K")
# plt.xlabel("Dihedral Angle (Degrees)")
# plt.ylabel("Count")
# plt.show()

twisted_atoms = []

frames = [236, 101, 118, 259, 363, 278, 483, 12, 254, 51]
for i, idx in enumerate(frames):
    coin_flip = np.random.randint(0, 2)
    dihedral_shift = np.random.normal(15, 3) if i < 5 else np.random.normal(-15, 3)
    print(f"dihedral shift: {dihedral_shift}")
    new_atoms = atoms_list[idx]
    new_atoms.set_dihedral(
        1, 5, 13, 14,
        new_atoms.get_dihedral(1, 5, 13, 14) + dihedral_shift,
        indices=[14, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26]
    )
    twisted_atoms.append(new_atoms)

for atom in twisted_atoms:
    view(atom)

write("C:/Users/alber/nequip/twisted-300K.xyz", twisted_atoms, format="extxyz")
#
# atoms = Atoms('HHCCHH', [[-1, 1, 0], [-1, -1, 0], [0, 0, 0],
#                          [1, 0, 0], [2, 1, 0], [2, -1, 0]])
# atoms.set_dihedral(1, 2, 3, 4, 30, mask=[0, 0, 0, 1, 1, 1])
# view(atoms)


# xyz = open("C:/Users/alber/nequip/configs_final/train_300K.xyz", "r")
# for idx, line in enumerate(xyz):
#     print(idx)
#     print(line)
# xyz.close()

# greater = np.where(test > 0.5)
# greater_elts = test[greater]
# print(greater)
# print(greater_elts)

# plt.plot(np.arange(5, 10, 0.5), test[:10], label="test10")
# plt.plot(test[10:])
# plt.legend(["test"])
# plt.show()

# perm = torch.randperm(500)
# print('[' + ', '.join(str(e.item()) for e in perm[:450]) + ']')
# print('[' + ', '.join(str(e.item()) for e in perm[450:500]) + ']')

# nums = np.arange(24)
# nums = nums.reshape((2, 3, -1))
# print(nums)
# var_nums = np.var(nums, axis=0)
# print(var_nums)
