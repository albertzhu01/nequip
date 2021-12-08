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
train_pred_energies = []
train_force_maes = []
train_pred_tot_e = []
train_tot_e_err = []

test_pred_forces = []
test_pred_energies = []
test_force_maes = []
test_pred_tot_e = []
test_tot_e_err = []

for i in range(10):
    # train_pred_forces.append(
    #     np.load(f"/n/home10/axzhu/nequip/ensembles_300K/train_forces_ensemble{i}.npz")['arr_0'])
    train_pred_energies.append(
        np.load(f"/n/home10/axzhu/nequip/ensembles_1200K/train_atomic_e_ensemble{i}_200K.npz")['arr_0'].reshape(-1))
    # train_force_maes.append(
    #     np.load(f"/n/home10/axzhu/nequip/ensembles_300K/train_forces_mae_ensemble{i}.npz")['arr_0'])
    train_pred_tot_e.append(
        np.load(f"/n/home10/axzhu/nequip/ensembles_1200K/train_tot_e_ensemble{i}_200K.npz")['arr_0'].reshape(-1))
    train_tot_e_err.append(
        np.load(f"/n/home10/axzhu/nequip/ensembles_1200K/train_tot_e_err_ensemble{i}_200K.npz")['arr_0'].reshape(-1))

    # test_pred_forces.append(
    #     np.load(f"/n/home10/axzhu/nequip/ensembles_300K/test_forces_ensemble{i}.npz")['arr_0'])
    test_pred_energies.append(
        np.load(f"/n/home10/axzhu/nequip/ensembles_1200K/test_atomic_e_ensemble{i}_200K.npz")['arr_0'].reshape(-1))
    # test_force_maes.append(
    #     np.load(f"/n/home10/axzhu/nequip/ensembles_300K/test_forces_mae_ensemble{i}.npz")['arr_0'])
    test_pred_tot_e.append(
        np.load(f"/n/home10/axzhu/nequip/ensembles_1200K/test_tot_e_ensemble{i}_200K.npz")['arr_0'].reshape(-1))
    test_tot_e_err.append(
        np.load(f"/n/home10/axzhu/nequip/ensembles_1200K/test_tot_e_err_ensemble{i}_200K.npz")['arr_0'].reshape(-1))

# for i in range(10, 20):
    # train_pred_forces.append(
    #     np.load(f"/n/home10/axzhu/nequip/ensembles_300K/train_forces_ensemble{i}_300K.npz")['arr_0'])
    # train_pred_energies.append(
    #     np.load(f"/n/home10/axzhu/nequip/ensembles_300K/train_atomic_e_ensemble{i}_300K.npz")['arr_0'].reshape(-1))
    # train_force_maes.append(
    #     np.load(f"/n/home10/axzhu/nequip/ensembles_300K/train_forces_mae_ensemble{i}_300K.npz")['arr_0'])
    # train_pred_tot_e.append(
    #     np.load(f"/n/home10/axzhu/nequip/ensembles_300K/train_tot_e_ensemble{i}_300K.npz")['arr_0'].reshape(-1))

    # test_pred_forces.append(
    #     np.load(f"/n/home10/axzhu/nequip/ensembles_300K/test_forces_ensemble{i}_300K.npz")['arr_0'])
    # test_pred_energies.append(
    #     np.load(f"/n/home10/axzhu/nequip/ensembles_300K/test_atomic_e_ensemble{i}_300K.npz")['arr_0'].reshape(-1))
    # test_force_maes.append(
    #     np.load(f"/n/home10/axzhu/nequip/ensembles_300K/test_forces_mae_ensemble{i}_300K.npz")['arr_0'])
    # test_pred_tot_e.append(
    #     np.load(f"/n/home10/axzhu/nequip/ensembles_300K/test_tot_e_ensemble{i}_300K.npz")['arr_0'].reshape(-1))



# train_pred_forces = np.array(train_pred_forces)
train_pred_energies = np.array(train_pred_energies)
# train_force_maes = np.array(train_force_maes)
train_pred_tot_e = np.array(train_pred_tot_e)
train_tot_e_err = np.array(train_tot_e_err)

# test_pred_forces = np.array(test_pred_forces)
test_pred_energies = np.array(test_pred_energies)
# test_force_maes = np.array(test_force_maes)
test_pred_tot_e = np.array(test_pred_tot_e)
test_tot_e_err = np.array(test_tot_e_err)

# print(f"train_pred_forces shape: {train_pred_forces.shape}")
print(f"train_pred_energies shape: {train_pred_energies.shape}")
# print(f"train_force_maes shape: {train_force_maes.shape}")
print(f"train_pred_tot_e shape: {train_pred_tot_e.shape}")
print(f"train_tot_e_err shape: {train_tot_e_err.shape}")

# print(f"test_pred_forces shape: {test_pred_forces.shape}")
print(f"test_pred_energies shape: {test_pred_energies.shape}")
# print(f"test_force_maes shape: {test_force_maes.shape}")
print(f"test_pred_tot_e shape: {test_pred_tot_e.shape}")
print(f"test_tot_e_err shape: {test_tot_e_err.shape}")

# var_train_forces = np.sum(np.var(train_pred_forces[0:10], axis=0), axis=1)
# max_var_train_forces = np.amax(var_train_forces.reshape(27, -1), axis=0)
var_train_energies = np.var(train_pred_energies, axis=0)
max_var_train_energies = np.amax(var_train_energies.reshape(27, -1), axis=0)
# mean_train_maes = np.mean(train_force_maes[0:10], axis=0)
var_train_tot_e = np.var(train_pred_tot_e, axis=0)
mean_train_tot_e_err = np.mean(train_tot_e_err, axis=0)

# var_test_forces = np.sum(np.var(test_pred_forces[0:10], axis=0), axis=1)
# max_var_test_forces = np.amax(var_test_forces.reshape(27, -1), axis=0)
var_test_energies = np.var(test_pred_energies, axis=0)
max_var_test_energies = np.amax(var_test_energies.reshape(27, -1), axis=0)
# mean_test_maes = np.mean(test_force_maes[0:10], axis=0)
var_test_tot_e = np.var(test_pred_tot_e, axis=0)
mean_test_tot_e_err = np.mean(test_tot_e_err, axis=0)

# print(f"var_train_forces shape: {var_train_forces.shape}")
# print(f"max_var_train_forces shape: {max_var_train_forces.shape}")
# print(f"var_train_energies shape: {var_train_energies.shape}")
print(f"max_var_train_forces shape: {max_var_train_energies.shape}")
# print(f"mean_train_maes shape: {mean_train_maes.shape}")
print(f"var_train_tot_e shape: {var_train_tot_e.shape}")
print(f"mean_train_tot_e_err shape: {mean_train_tot_e_err.shape}")

# print(f"var_test_forces shape: {var_test_forces.shape}")
# print(f"max_var_test_forces shape: {max_var_test_forces.shape}")
# print(f"var_test_energies shape: {var_test_energies.shape}")
print(f"max_var_test_forces shape: {max_var_test_energies.shape}")
# print(f"mean_test_maes shape: {mean_test_maes.shape}")
print(f"var_test_tot_e shape: {var_test_tot_e.shape}")
print(f"mean_test_tot_e_err shape: {mean_test_tot_e_err.shape}")

# tot_test_atoms = len(var_test_forces)
# tot_train_atoms = len(var_train_forces)
# num_bpa_atoms = 27
# mae_cutoff = 0.043

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

# Maximum Atomic Energy Variance vs. Total Energy MAE (or variance)
plt.figure()
plt.subplots(figsize=(16, 9))
plt.rc('xtick', labelsize=14)
plt.rc('ytick', labelsize=14)
plt.scatter(
    x=var_train_tot_e,
    y=max_var_train_energies,
    color='k',
    label=f'Training Data'
)
plt.scatter(
    x=var_test_tot_e,
    y=max_var_test_energies,
    color='b',
    label=f'Test Data'
)
plt.legend(fontsize=16)
plt.title(
    f"Max. Atomic Energy Variance vs. Total Energy Variance (Train 300K, Test 1200K)",
    fontsize=24
)
plt.xlabel("Total Energy Variance (eV²)", fontsize=20)
plt.ylabel("Maximum Local Energy Variance (eV²)", fontsize=20)
plt.savefig(f"loc-e-var_tot-var_1200K.png")

# Total Energy Variance vs. Total Energy Squared MAE
# plt.figure()
# plt.subplots(figsize=(16, 9))
# plt.rc('xtick', labelsize=14)
# plt.rc('ytick', labelsize=14)
# plt.scatter(
#     x=np.square(mean_train_tot_e_err),
#     y=var_train_tot_e,
#     color='k',
#     label=f'Training Data'
# )
# plt.scatter(
#     x=np.square(mean_test_tot_e_err),
#     y=var_test_tot_e,
#     color='b',
#     label=f'Test Data'
# )
# plt.legend(fontsize=16)
# plt.title(
#     f"Total Energy Variance vs. Total Energy Squared MAE (Train 300K, Test 300K)",
#     fontsize=24
# )
# plt.xlabel("Total Energy Squared MAE (eV²)", fontsize=20)
# plt.ylabel("Total Energy Variance (eV²)", fontsize=20)
# plt.savefig(f"tot-e_var_err_300K.png")

# for i in range(num_bpa_atoms):

    # Energy variance vs energy mae
    # plt.figure()
    # plt.subplots(figsize=(16, 9))
    # plt.rc('xtick', labelsize=14)
    # plt.rc('ytick', labelsize=14)
    # plt.scatter(
    #     x=mean_train_maes[i:12150:27],
    #     y=var_train_energies[i:12150:27],
    #     color='k',
    #     label=f'Training Data'
    # )
    # plt.scatter(
    #     x=mean_test_maes[i:48735:27],
    #     y=var_test_energies[i:48735:27],
    #     color='b',
    #     label=f'Test Data'
    # )
    # plt.legend(fontsize=14)
    # plt.title(
    #     f"Atom Index {i} Predicted Atomic Energy Variance vs. Force MAE (Train 300K, Test 1200K)",
    #     fontsize=18
    # )
    # plt.xlabel("Force MAE (eV/A)", fontsize=16)
    # plt.ylabel("Variance of Predicted Atomic Energies (eV²)", fontsize=16)
    # plt.savefig(f"atom{i}_e-var_vs_mae_1200K.png")

    # Force variance vs. energy mae
    # atom_i_maes = mean_test_maes[i:tot_test_atoms:num_bpa_atoms]
    # atom_i_var_forces = var_test_forces[i:tot_test_atoms:num_bpa_atoms]
    # bad_test_data_idxs = np.where(atom_i_maes > mae_cutoff)
    # bad_mean_maes = atom_i_maes[bad_test_data_idxs]
    # bad_force_vars = atom_i_var_forces[bad_test_data_idxs]
    #
    # good_test_data_idxs = np.where(atom_i_maes <= mae_cutoff)
    # good_force_vars = atom_i_var_forces[good_test_data_idxs]
    # var_cutoff = np.amax(atom_i_var_forces) if len(bad_force_vars) == 0 else np.amin(bad_force_vars)
    #
    # true_pos = len(bad_mean_maes)
    # false_pos = len(good_force_vars[np.where(good_force_vars > var_cutoff)])
    #
    # plt.figure()
    # plt.subplots(figsize=(16, 9))
    # plt.rc('xtick', labelsize=16)
    # plt.rc('ytick', labelsize=16)
    # plt.scatter(
    #     x=np.square(mean_train_maes[i:tot_train_atoms:num_bpa_atoms]),
    #     y=var_train_forces[i:tot_train_atoms:num_bpa_atoms],
    #     color='k',
    #     label=f'Training Data'
    # )
    # plt.scatter(
    #     x=np.square(atom_i_maes),
    #     y=atom_i_var_forces,
    #     color='b',
    #     label=f'Good Test Data: {false_pos} / {tot_test_atoms // num_bpa_atoms} (false positives / total data points)'
    # )
    # plt.scatter(
    #     x=np.square(bad_mean_maes),
    #     y=bad_force_vars,
    #     color='r',
    #     label=f'Bad Test Data: {true_pos} / {tot_test_atoms // num_bpa_atoms} (true positives / total data points)'
    # )
    # plt.axhline(
    #     var_cutoff,
    #     color='k',
    #     linestyle='--',
    #     label='Variance cutoff (min variance of bad test data)'
    # )
    # plt.axvline(mae_cutoff ** 2, color='m', linestyle='--', label='Chemical accuracy cutoff')
    # plt.legend(fontsize=16)
    # plt.title(
    #     f"Atom Index {i} Atomic Force Variance vs. Force MSE (Train 300K, Test 300K)",
    #     fontsize=24
    # )
    # plt.xlabel("Force MSE ((eV/A)²)", fontsize=20)
    # plt.ylabel("Variance of Predicted Atomic Forces ((eV/A)²)", fontsize=20)
    # plt.savefig(f"atom{i}_f-var_vs_mae_300K.png")
