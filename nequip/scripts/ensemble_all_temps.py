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
train_force_maes = []

test_pred_forces3 = []
test_force_maes3 = []

test_pred_forces6 = []
test_force_maes6 = []

test_pred_forces12 = []
test_force_maes12 = []

for i in range(10):
    train_pred_forces.append(
        np.load(f"/n/home10/axzhu/nequip/ensembles_300K/train_forces_ensemble{i}.npz")['arr_0'])
    train_force_maes.append(
        np.load(f"/n/home10/axzhu/nequip/ensembles_300K/train_forces_mae_ensemble{i}.npz")['arr_0'])

    test_pred_forces3.append(
        np.load(f"/n/home10/axzhu/nequip/ensembles_300K/test_forces_ensemble{i}.npz")['arr_0'])
    test_force_maes3.append(
        np.load(f"/n/home10/axzhu/nequip/ensembles_300K/test_forces_mae_ensemble{i}.npz")['arr_0'])

    test_pred_forces6.append(
        np.load(f"/n/home10/axzhu/nequip/ensembles_600K/test_forces_ensemble{i}_600K.npz")['arr_0'])
    test_force_maes6.append(
        np.load(f"/n/home10/axzhu/nequip/ensembles_600K/test_forces_mae_ensemble{i}_600K.npz")['arr_0'])

    test_pred_forces12.append(
        np.load(f"/n/home10/axzhu/nequip/ensembles_1200K/test_forces_ensemble{i}_200K.npz")['arr_0'])
    test_force_maes12.append(
        np.load(f"/n/home10/axzhu/nequip/ensembles_1200K/test_forces_mae_ensemble{i}_200K.npz")['arr_0'])

train_pred_forces = np.array(train_pred_forces)
train_force_maes = np.array(train_force_maes)

test_pred_forces3 = np.array(test_pred_forces3)
test_force_maes3 = np.array(test_force_maes3)

test_pred_forces6 = np.array(test_pred_forces6)
test_force_maes6 = np.array(test_force_maes6)

test_pred_forces12 = np.array(test_pred_forces12)
test_force_maes12 = np.array(test_force_maes12)

print(f"train_pred_forces shape: {train_pred_forces.shape}")
print(f"train_force_maes shape: {train_force_maes.shape}")

print(f"test_pred_forces shape: {test_pred_forces3.shape}")
print(f"test_force_maes shape: {test_force_maes3.shape}")

var_train_forces = np.sum(np.var(train_pred_forces, axis=0), axis=1)
mean_train_maes = np.mean(train_force_maes, axis=0)

var_test_forces3 = np.sum(np.var(test_pred_forces3, axis=0), axis=1)
mean_test_maes3 = np.mean(test_force_maes3, axis=0)

var_test_forces6 = np.sum(np.var(test_pred_forces6, axis=0), axis=1)
mean_test_maes6 = np.mean(test_force_maes6, axis=0)

var_test_forces12 = np.sum(np.var(test_pred_forces12, axis=0), axis=1)
mean_test_maes12 = np.mean(test_force_maes12, axis=0)

print(f"var_train_forces shape: {var_train_forces.shape}")
print(f"mean_train_maes shape: {mean_train_maes.shape}")

print(f"var_test_forces shape: {var_test_forces3.shape}")
print(f"mean_test_maes shape: {mean_test_maes3.shape}")

tot_test_atoms3 = len(var_test_forces3)
tot_test_atoms6 = len(var_test_forces6)
tot_test_atoms12 = len(var_test_forces12)
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

for i in range(27):

    plt.figure()
    plt.subplots(figsize=(16, 9))
    plt.rc('xtick', labelsize=16)
    plt.rc('ytick', labelsize=16)
    plt.scatter(
        x=mean_test_maes12[i:tot_test_atoms12:num_bpa_atoms],
        y=var_test_forces12[i:tot_test_atoms12:num_bpa_atoms],
        color='r',
        label=f'Test Data 1200K'
    )
    plt.scatter(
        x=mean_test_maes6[i:tot_test_atoms6:num_bpa_atoms],
        y=var_test_forces6[i:tot_test_atoms6:num_bpa_atoms],
        color='g',
        label=f'Test Data 600K'
    )
    plt.scatter(
        x=mean_test_maes3[i:tot_test_atoms3:num_bpa_atoms],
        y=var_test_forces3[i:tot_test_atoms3:num_bpa_atoms],
        color='b',
        label=f'Test Data 300K'
    )
    plt.scatter(
        x=mean_train_maes[i:tot_train_atoms:num_bpa_atoms],
        y=var_train_forces[i:tot_train_atoms:num_bpa_atoms],
        color='k',
        label=f'Training Data 300K'
    )

    plt.legend(fontsize=16)
    plt.title(
        f"Atom Index {i} Predicted Atomic Force Variance vs. Force MAE",
        fontsize=24
    )
    plt.xlabel("Force MAE (eV/A)", fontsize=20)
    plt.ylabel("Variance of Predicted Atomic Forces ((eV/A)^2)", fontsize=20)
    plt.savefig(f"atom{i}_f-var_vs_mae.png")
