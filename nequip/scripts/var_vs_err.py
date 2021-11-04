import torch
import numpy as np
import matplotlib.pyplot as plt
import argparse
import textwrap

from pathlib import Path
from sklearn.metrics import mean_absolute_error
from nequip.utils import Config, dataset_from_config
from nequip.data import AtomicDataDict, AtomicData, Collater


# Script for plotting variances vs. force MAE by loading data outputted from ensemble_uncertainty.py
train_pred_forces = []
# train_pred_energies = []
# train_force_maes = []
train_pred_tot_e = []

test_pred_forces = []
# test_pred_energies = []
# test_force_maes = []
test_pred_tot_e = []
for i in range(10):
    train_pred_forces.append(
        np.load(f"/n/home10/axzhu/nequip/ensembles_600K/train_forces_ensemble{i}_600K.npz")['arr_0'])
    # train_pred_energies.append(
    #     np.load(f"/n/home10/axzhu/nequip/ensembles_600K/train_atomic_e_ensemble{i}_600K.npz")['arr_0'].reshape(-1))
    # train_force_maes.append(
    #     np.load(f"/n/home10/axzhu/nequip/ensembles_600K/train_forces_mae_ensemble{i}_600K.npz")['arr_0'])
    train_pred_tot_e.append(
        np.load(f"/n/home10/axzhu/nequip/ensembles_600K/train_tot_e_ensemble{i}_600K.npz")['arr_0'].reshape(-1))

    test_pred_forces.append(
        np.load(f"/n/home10/axzhu/nequip/ensembles_600K/test_forces_ensemble{i}_600K.npz")['arr_0'])
    # test_pred_energies.append(
    #     np.load(f"/n/home10/axzhu/nequip/ensembles_600K/test_atomic_e_ensemble{i}_600K.npz")['arr_0'].reshape(-1))
    # test_force_maes.append(
    #     np.load(f"/n/home10/axzhu/nequip/ensembles_600K/test_forces_mae_ensemble{i}_600K.npz")['arr_0'])
    test_pred_tot_e.append(
        np.load(f"/n/home10/axzhu/nequip/ensembles_600K/test_tot_e_ensemble{i}_600K.npz")['arr_0'].reshape(-1))

train_pred_forces = np.array(train_pred_forces)
# train_pred_energies = np.array(train_pred_energies)
# train_force_maes = np.array(train_force_maes)
train_pred_tot_e = np.array(train_pred_tot_e)

test_pred_forces = np.array(test_pred_forces)
# test_pred_energies = np.array(test_pred_energies)
# test_force_maes = np.array(test_force_maes)
test_pred_tot_e = np.array(test_pred_tot_e)

print(f"train_pred_forces shape: {train_pred_forces.shape}")
# print(f"train_pred_energies shape: {train_pred_energies.shape}")
# print(f"train_force_maes shape: {train_force_maes.shape}")
print(f"train_pred_tot_e shape: {train_pred_tot_e.shape}")

print(f"test_pred_forces shape: {test_pred_forces.shape}")
# print(f"test_pred_energies shape: {test_pred_energies.shape}")
# print(f"test_force_maes shape: {test_force_maes.shape}")
print(f"test_pred_tot_e shape: {test_pred_tot_e.shape}")

var_train_forces = np.sum(np.var(train_pred_forces, axis=0), axis=1)
max_var_train_forces = np.amax(var_train_forces.reshape(27, -1), axis=0)
# var_train_energies = np.var(train_pred_energies, axis=0)
# mean_train_maes = np.mean(train_force_maes, axis=0)
var_train_tot_e = np.var(train_pred_tot_e, axis=0)

var_test_forces = np.sum(np.var(test_pred_forces, axis=0), axis=1)
max_var_test_forces = np.amax(var_test_forces.reshape(27, -1), axis=0)
# var_test_energies = np.var(test_pred_energies, axis=0)
# mean_test_maes = np.mean(test_force_maes, axis=0)
var_test_tot_e = np.var(test_pred_tot_e, axis=0)

print(f"var_train_forces shape: {var_train_forces.shape}")
print(f"max_var_train_forces shape: {max_var_train_forces.shape}")
# print(f"var_train_energies shape: {var_train_energies.shape}")
# print(f"mean_train_maes shape: {mean_train_maes.shape}")
print(f"var_train_tot_e shape: {var_train_tot_e.shape}")

print(f"var_test_forces shape: {var_test_forces.shape}")
print(f"max_var_test_forces shape: {max_var_test_forces.shape}")
# print(f"var_test_energies shape: {var_test_energies.shape}")
# print(f"mean_test_maes shape: {mean_test_maes.shape}")
print(f"var_test_tot_e shape: {var_test_tot_e.shape}")

# Maximum Atomic Force Variance vs. Total Energy Variance
plt.figure()
plt.subplots(figsize=(16, 9))
plt.rc('xtick', labelsize=14)
plt.rc('ytick', labelsize=14)
plt.scatter(
    x=var_train_tot_e,
    y=max_var_train_forces,
    color='k',
    label=f'Training Data'
)
plt.scatter(
    x=var_test_tot_e,
    y=max_var_test_forces,
    color='b',
    label=f'Test Data'
)
plt.legend(fontsize=14)
plt.title(
    f"Max Atomic Force Variance vs. Total Energy Variance (Train 300K, Test 600K)",
    fontsize=18
)
plt.xlabel("Total Energy Variance (eV^2)", fontsize=16)
plt.ylabel("Max Atomic Force Variance ((eV/A)^2)", fontsize=16)
plt.savefig(f"tot-e-var_vs_f-var_600K.png")

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
#         f"Atom Index {i} Predicted Atomic Energy Variance vs. Force MAE (Train 300K, Test 1200K)",
#         fontsize=18
#     )
#     plt.xlabel("Force MAE (eV/A)", fontsize=16)
#     plt.ylabel("Variance of Predicted Atomic Energies (eV^2)", fontsize=16)
#     plt.savefig(f"atom{i}_e-var_vs_mae_1200K.png")

    # Force variance vs. energy mae
    # plt.figure()
    # plt.subplots(figsize=(16, 9))
    # plt.rc('xtick', labelsize=14)
    # plt.rc('ytick', labelsize=14)
    # plt.scatter(
    #     x=mean_train_maes[i:12150:27],
    #     y=var_train_forces[i:12150:27],
    #     color='k',
    #     label=f'Training Data'
    # )
    # plt.scatter(
    #     x=mean_test_maes[i:48735:27],
    #     y=var_test_forces[i:48735:27],
    #     color='b',
    #     label=f'Test Data'
    # )
    # plt.legend(fontsize=14)
    # plt.title(
    #     f"Atom Index {i} Predicted Atomic Force Variance vs. Force MAE (Train 300K, Test 1200K)",
    #     fontsize=18
    # )
    # plt.xlabel("Force MAE (eV/A)", fontsize=16)
    # plt.ylabel("Variance of Predicted Atomic Forces ((eV/A)^2)", fontsize=16)
    # plt.savefig(f"atom{i}_f-var_vs_mae_1200K.png")
