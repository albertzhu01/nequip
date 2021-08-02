import torch
import numpy as np
from sklearn import mixture
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from ase.visualize import view
from sklearn.metrics import mean_absolute_error
from scipy import stats

from nequip.utils import Config, dataset_from_config
from nequip.data import AtomicDataDict, AtomicData, Collater
from nequip.nn import SequentialGraphNetwork, SaveForOutput

f, ax = plt.subplots(figsize=(19, 9.5))

# path = "C:/Users/alber/nequip/nequip/scripts/aspirin_50_epochs_new/results/aspirin/example-run"
path = "/n/home10/axzhu/nequip/results/bpa/train300K_072321"

model = torch.load(path + "/best_model.pth", map_location=torch.device('cpu'))
model.eval()

# Load a config file
config = Config.from_file(path + "/config_final.yaml")
dataset = dataset_from_config(config)
# config_test = Config.from_file("C:/Users/alber/nequip/nequip/scripts/aspirin_50_epochs_new/results/aspirin/example-run/config_final.yaml")
config_test = Config.from_file("/n/home10/axzhu/nequip/configs/dataset.yaml")
dataset_test = dataset_from_config(config_test)

# Sample two sets of 100 points from the test set
sample_idxs = torch.randperm(len(dataset_test))[:200]
sample1 = sample_idxs[:100]
print(sample1)
sample2 = sample_idxs[100:]
print(sample2)
test_sample1 = [dataset_test.get(idx.item()) for idx in sample1]
test_sample2 = [dataset_test.get(idx.item()) for idx in sample2]

# Evaluate model on test data samples and extract atomic positions and features
c_test = Collater.for_dataset(dataset_test, exclude_keys=[])
test_batch1 = c_test.collate(test_sample1)
test_batch2 = c_test.collate(test_sample2)
print("Begin model evaluation on test data...")
test_out1 = model(AtomicData.to_AtomicDataDict(test_batch1))
test_out2 = model(AtomicData.to_AtomicDataDict(test_batch2))

test_pos1 = test_out1[AtomicDataDict.POSITIONS_KEY].detach().numpy()
test_pos2 = test_out2[AtomicDataDict.POSITIONS_KEY].detach().numpy()
print(f"Atomic positions shape: {test_pos1.shape}")

test_features1 = test_out1[AtomicDataDict.NODE_FEATURES_KEY]
test_features2 = test_out2[AtomicDataDict.NODE_FEATURES_KEY]
print(f"Atomic features shape: {test_features1.shape}")

# Get dimensions of train and test features and number of atoms in molecule
test_sample_len = len(test_sample1)
test_tot_atoms, feature_length = test_features1.shape
num_atoms = test_tot_atoms // test_sample_len
print(f"Total test atoms per sample: {test_tot_atoms}")
print(f"Number of atoms in molecule: {num_atoms}")

# First compute pairwise distances between atoms for each molecule in the 2 test samples
pos_dists1 = np.zeros((num_atoms, test_sample_len, num_atoms))
pos_dists2 = np.zeros((num_atoms, test_sample_len, num_atoms))
for i in range(len(test_sample1)):
    for j in range(num_atoms):
        for k in range(j, num_atoms):
            atom_j1_pos = test_pos1[i * num_atoms + j]
            atom_k1_pos = test_pos1[i * num_atoms + k]
            pos_dists1[j][i][k] = np.linalg.norm(atom_k1_pos - atom_j1_pos)
            pos_dists1[k][i][j] = pos_dists1[j][i][k]

            atom_j2_pos = test_pos2[i * num_atoms + j]
            atom_k2_pos = test_pos2[i * num_atoms + k]
            pos_dists2[j][i][k] = np.linalg.norm(atom_k2_pos - atom_j2_pos)
            pos_dists2[k][i][j] = pos_dists2[j][i][k]

print(f"Atom distances shape: {pos_dists2.shape}")

# Next compute the atomic input and feature distances per atom for all 100 molecules in the 2 test samples
for i in range(1):
    atom_i_dists1 = torch.tensor(pos_dists1[i:i+1])
    atom_i_dists2 = torch.tensor(pos_dists2[i:i+1])
    print(f"Atomic distances shape: {atom_i_dists1.shape}")
    input_dists = torch.cdist(atom_i_dists1, atom_i_dists2, p=2).view(-1)

    print(f"Input distances shape: {input_dists.shape}")

    feature_dists = torch.cdist(
        test_features1[i:test_tot_atoms:num_atoms].view(1, test_sample_len, feature_length),
        test_features2[i:test_tot_atoms:num_atoms].view(1, test_sample_len, feature_length),
        p=2
    ).view(-1)

    print(f"Feature distances shape: {feature_dists.shape}")

    # Plot
    plt.figure()
    plt.subplots(figsize=(16, 9))
    plt.scatter(
        x=input_dists.detach().numpy(),
        y=feature_dists.detach().numpy(),
    )
    plt.title(f"3BPA Atom Index {i} Feature Distance vs. Input Distance (300K Test)")
    plt.xlabel("Input Distance (A)")
    plt.ylabel("Feature Distance")
    plt.savefig(f"bpa_atom{i}_i-f-dist_300K.png")
