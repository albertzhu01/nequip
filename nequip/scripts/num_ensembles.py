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
train_force_maes = []
# train_pred_tot_e = []

test_pred_forces = []
# test_pred_energies = []
test_force_maes = []
# test_pred_tot_e = []
# for i in range(10):
#     train_pred_forces.append(
#         np.load(f"/n/home10/axzhu/nequip/ensembles_300K/train_forces_ensemble{i}.npz")['arr_0'])
#     # train_pred_energies.append(
#     #     np.load(f"/n/home10/axzhu/nequip/ensembles_300K/train_atomic_e_ensemble{i}_300K.npz")['arr_0'].reshape(-1))
#     train_force_maes.append(
#         np.load(f"/n/home10/axzhu/nequip/ensembles_300K/train_forces_mae_ensemble{i}.npz")['arr_0'])
#     # train_pred_tot_e.append(
#     #     np.load(f"/n/home10/axzhu/nequip/ensembles_300K/train_tot_e_ensemble{i}_300K.npz")['arr_0'].reshape(-1))
#
#     test_pred_forces.append(
#         np.load(f"/n/home10/axzhu/nequip/ensembles_300K/test_forces_ensemble{i}.npz")['arr_0'])
#     # test_pred_energies.append(
#     #     np.load(f"/n/home10/axzhu/nequip/ensembles_300K/test_atomic_e_ensemble{i}_300K.npz")['arr_0'].reshape(-1))
#     test_force_maes.append(
#         np.load(f"/n/home10/axzhu/nequip/ensembles_300K/test_forces_mae_ensemble{i}.npz")['arr_0'])
#     # test_pred_tot_e.append(
#     #     np.load(f"/n/home10/axzhu/nequip/ensembles_300K/test_tot_e_ensemble{i}_300K.npz")['arr_0'].reshape(-1))

for i in range(20):
    train_pred_forces.append(
        np.load(f"/n/home10/axzhu/nequip/ensembles_1200K/train_forces_ensemble{i}_600K.npz")['arr_0'])
    # train_pred_energies.append(
    #     np.load(f"/n/home10/axzhu/nequip/ensembles_300K/train_atomic_e_ensemble{i}_300K.npz")['arr_0'].reshape(-1))
    train_force_maes.append(
        np.load(f"/n/home10/axzhu/nequip/ensembles_1200K/train_forces_mae_ensemble{i}_600K.npz")['arr_0'])
    # train_pred_tot_e.append(
    #     np.load(f"/n/home10/axzhu/nequip/ensembles_300K/train_tot_e_ensemble{i}_300K.npz")['arr_0'].reshape(-1))

    test_pred_forces.append(
        np.load(f"/n/home10/axzhu/nequip/ensembles_1200K/test_forces_ensemble{i}_600K.npz")['arr_0'])
    # test_pred_energies.append(
    #     np.load(f"/n/home10/axzhu/nequip/ensembles_300K/test_atomic_e_ensemble{i}_300K.npz")['arr_0'].reshape(-1))
    test_force_maes.append(
        np.load(f"/n/home10/axzhu/nequip/ensembles_1200K/test_forces_mae_ensemble{i}_600K.npz")['arr_0'])
    # test_pred_tot_e.append(
    #     np.load(f"/n/home10/axzhu/nequip/ensembles_300K/test_tot_e_ensemble{i}_300K.npz")['arr_0'].reshape(-1))

train_pred_forces = np.array(train_pred_forces)
# train_pred_energies = np.array(train_pred_energies)
train_force_maes = np.array(train_force_maes)
# train_pred_tot_e = np.array(train_pred_tot_e)

test_pred_forces = np.array(test_pred_forces)
# test_pred_energies = np.array(test_pred_energies)
test_force_maes = np.array(test_force_maes)
# test_pred_tot_e = np.array(test_pred_tot_e)

print(f"train_pred_forces shape: {train_pred_forces.shape}")
# print(f"train_pred_energies shape: {train_pred_energies.shape}")
print(f"train_force_maes shape: {train_force_maes.shape}")
# print(f"train_pred_tot_e shape: {train_pred_tot_e.shape}")

print(f"test_pred_forces shape: {test_pred_forces.shape}")
# print(f"test_pred_energies shape: {test_pred_energies.shape}")
print(f"test_force_maes shape: {test_force_maes.shape}")
# print(f"test_pred_tot_e shape: {test_pred_tot_e.shape}")

var_train_forces = np.sum(np.var(train_pred_forces[0:5], axis=0), axis=1)
max_var_train_forces = np.amax(var_train_forces.reshape(27, -1), axis=0)
# var_train_energies = np.var(train_pred_energies, axis=0)
mean_train_maes = np.mean(train_force_maes[0:5], axis=0)
# var_train_tot_e = np.var(train_pred_tot_e, axis=0)

var_test_forces5 = np.sum(np.var(test_pred_forces[0:5], axis=0), axis=1)
max_var_test_forces5 = np.amax(var_test_forces5.reshape(27, -1), axis=0)
# var_test_energies = np.var(test_pred_energies, axis=0)
mean_test_maes5 = np.mean(test_force_maes[0:5], axis=0)
# var_test_tot_e = np.var(test_pred_tot_e, axis=0)

print(f"var_train_forces shape: {var_train_forces.shape}")
print(f"max_var_train_forces shape: {max_var_train_forces.shape}")
# print(f"var_train_energies shape: {var_train_energies.shape}")
print(f"mean_train_maes shape: {mean_train_maes.shape}")
# print(f"var_train_tot_e shape: {var_train_tot_e.shape}")

print(f"var_test_forces shape: {var_test_forces5.shape}")
print(f"max_var_test_forces shape: {max_var_test_forces5.shape}")
# print(f"var_test_energies shape: {var_test_energies.shape}")
print(f"mean_test_maes shape: {mean_test_maes5.shape}")
# print(f"var_test_tot_e shape: {var_test_tot_e.shape}")

var_test_forces10 = np.sum(np.var(test_pred_forces[0:10], axis=0), axis=1)
max_var_test_forces10 = np.amax(var_test_forces10.reshape(27, -1), axis=0)
# var_test_energies = np.var(test_pred_energies, axis=0)
mean_test_maes10 = np.mean(test_force_maes[0:10], axis=0)
# var_test_tot_e = np.var(test_pred_tot_e, axis=0)

var_test_forces15 = np.sum(np.var(test_pred_forces[0:15], axis=0), axis=1)
max_var_test_forces15 = np.amax(var_test_forces15.reshape(27, -1), axis=0)
# var_test_energies = np.var(test_pred_energies, axis=0)
mean_test_maes15 = np.mean(test_force_maes[0:15], axis=0)
# var_test_tot_e = np.var(test_pred_tot_e, axis=0)

var_test_forces20 = np.sum(np.var(test_pred_forces, axis=0), axis=1)
max_var_test_forces20 = np.amax(var_test_forces20.reshape(27, -1), axis=0)
# var_test_energies = np.var(test_pred_energies, axis=0)
mean_test_maes20 = np.mean(test_force_maes, axis=0)
# var_test_tot_e = np.var(test_pred_tot_e, axis=0)

tot_test_atoms = len(var_test_forces5)
tot_train_atoms = len(var_train_forces)
num_bpa_atoms = 27
mae_cutoff = 0.043

# Maximum Atomic Force Variance vs. Total Energy Variance
# plt.figure()
# plt.subplots(figsize=(16, 9))
# plt.rc('xtick', labelsize=14)
# plt.rc('ytick', labelsize=14)
# plt.scatter(
#     x=var_train_tot_e,
#     y=max_var_train_forces,
#     color='k',
#     label=f'Training Data'
# )
# plt.scatter(
#     x=var_test_tot_e,
#     y=max_var_test_forces,
#     color='b',
#     label=f'Test Data'
# )
# plt.legend(fontsize=14)
# plt.title(
#     f"Max Atomic Force Variance vs. Total Energy Variance (Train 300K, Test 300K)",
#     fontsize=18
# )
# plt.xlabel("Total Energy Variance (eV^2)", fontsize=16)
# plt.ylabel("Max Atomic Force Variance ((eV/A)^2)", fontsize=16)
# plt.savefig(f"tot-e-var_vs_f-var_300K.png")

for i in range(num_bpa_atoms):
    plt.figure()
    plt.subplots(figsize=(16, 9))
    plt.rc('xtick', labelsize=16)
    plt.rc('ytick', labelsize=16)
    plt.scatter(
        x=mean_test_maes5[i:tot_test_atoms:num_bpa_atoms],
        y=var_test_forces5[i:tot_test_atoms:num_bpa_atoms],
        color='r',
        label=f'5 ensembles'
    )
    plt.scatter(
        x=mean_test_maes10[i:tot_test_atoms:num_bpa_atoms],
        y=var_test_forces10[i:tot_test_atoms:num_bpa_atoms],
        color='g',
        label=f'10 ensembles'
    )
    plt.scatter(
        x=mean_test_maes15[i:tot_test_atoms:num_bpa_atoms],
        y=var_test_forces15[i:tot_test_atoms:num_bpa_atoms],
        color='b',
        label=f'15 ensembles'
    )
    plt.scatter(
        x=mean_test_maes20[i:tot_test_atoms:num_bpa_atoms],
        y=var_test_forces20[i:tot_test_atoms:num_bpa_atoms],
        color='k',
        label=f'20 ensembles'
    )

    plt.legend(fontsize=16)
    plt.title(
        f"Atom Index {i} Atomic Force Stddev vs. Force MAE (Train 300K, Test 600K)",
        fontsize=24
    )
    plt.xlabel("Force MAE (eV/A)", fontsize=20)
    plt.ylabel("Standard Deviation of Predicted Atomic Forces ((eV/A)²)", fontsize=20)
    plt.savefig(f"atom{i}_num_ensem_stddev_600K.png")
