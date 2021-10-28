import torch
import numpy as np
import matplotlib.pyplot as plt
import argparse
import textwrap

from pathlib import Path
from sklearn.metrics import mean_absolute_error
from nequip.utils import Config, dataset_from_config
from nequip.data import AtomicDataDict, AtomicData, Collater

train_pred_forces = []
train_pred_energies = []
train_force_maes = []

test_pred_forces = []
test_pred_energies = []
test_force_maes = []
for i in range(1):
    train_pred_forces.append(np.load(f"/n/home10/axzhu/nequip/ensembles/train_forces_ensemble{i}")['arr_0'])
    train_pred_energies.append(
        np.load(f"/n/home10/axzhu/nequip/ensembles/train_atomic_e_ensemble{i}")['arr_0'].reshape(-1))
    train_force_maes.append(np.load(f"/n/home10/axzhu/nequip/ensembles/train_forces_mae_ensemble{i}")['arr_0'])

    test_pred_forces.append(np.load(f"/n/home10/axzhu/nequip/ensembles/test_forces_ensemble{i}")['arr_0'])
    test_pred_energies.append(
        np.load(f"/n/home10/axzhu/nequip/ensembles/test_atomic_e_ensemble{i}")['arr_0'].reshape(-1))
    test_force_maes.append(np.load(f"/n/home10/axzhu/nequip/ensembles/test_forces_mae_ensemble{i}")['arr_0'])

train_pred_forces = np.array(train_pred_forces)
train_pred_energies = np.array(train_pred_energies)
train_force_maes = np.array(train_force_maes)

test_pred_forces = np.array(test_pred_forces)
test_pred_energies = np.array(test_pred_energies)
test_force_maes = np.array(test_force_maes)

print(f"train_pred_forces shape: {train_pred_forces.shape}")
print(f"train_pred_energies shape: {train_pred_forces.shape}")
print(f"train_force_maes shape: {train_pred_forces.shape}")
print(f"test_pred_forces shape: {train_pred_forces.shape}")
print(f"test_pred_energies shape: {train_pred_forces.shape}")
print(f"test_force_maes shape: {train_pred_forces.shape}")

# var_train_forces = np.var(train_pred_forces, axis=0)
# var_train_energies = np.var(train_pred_energies, axis=0)
# mean_train_maes = np.mean(train_force_maes, axis=0)

# var_test_forces = np.var(test_pred_forces, axis=0)
# var_test_energies = np.var(test_pred_energies, axis=0)
# mean_test_maes = np.mean(test_force_maes, axis=0)

# for i in range(27):
#
#     # Energy variance vs energy mae
#     plt.figure()
#     plt.subplots(figsize=(16, 9))
#     plt.rc('xtick', labelsize=14)
#     plt.rc('ytick', labelsize=14)
#     plt.scatter(
#         x=mean_train_maes[i:12150:27],
#         y=var_train_energies[i:12150:27],
#         color='k',
#         label=f'Training Data'
#     )
#     plt.scatter(
#         x=mean_test_maes[i:48735:27],
#         y=var_test_energies[i:48735:27],
#         color='b',
#         label=f'Test Data'
#     )
#     plt.legend(fontsize=14)
#     plt.title(
#         f"Atom Index {i} Predicted Atomic Energy Variance vs. Force MAE (Train 300K, Test 300K)",
#         fontsize=18
#     )
#     plt.xlabel("Force MAE (eV/A)", fontsize=16)
#     plt.ylabel("Variance of Predicted Atomic Energies (eV^2)", fontsize=16)
#     plt.savefig(f"atom{i}_e-var_vs_mae_300K.png")

    # Force variance vs. energy mae
    # plt.figure()
    # plt.subplots(figsize=(16, 9))
    # plt.rc('xtick', labelsize=14)
    # plt.rc('ytick', labelsize=14)
    # plt.scatter(
    #     x=mean_train_maes,
    #     y=var_train_energies,
    #     color='k',
    #     label=f'Training Data'
    # )
    # plt.scatter(
    #     x=mean_test_maes,
    #     y=var_test_energies,
    #     color='b',
    #     label=f'Test Data'
    # )
    # plt.legend(fontsize=14)
    # plt.title(
    #     f"Atom Index {i} Predicted Atomic Energy Variance vs. Force MAE (Train 300K, Test 300K)",
    #     fontsize=18
    # )
    # plt.xlabel("Force MAE (eV/A)", fontsize=16)
    # plt.ylabel("Variance of Predicted Atomic Energies (eV^2)", fontsize=16)
    # plt.savefig(f"atom{i}_e-var_vs_mae_300K.png")

