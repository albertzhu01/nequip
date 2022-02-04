import torch
import numpy as np
from sklearn import mixture
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from ase.visualize import view
from ase.io import read
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

atoms = read("C:/Users/alber/nequip/bpasub-no-train-600K.xyz", index=":", format="extxyz")
for idx, atom in enumerate(atoms[:2]):
    print(atom)
    print(atom.get_potential_energies())
    for key in atom:
        print(key)
# print(atoms)
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
