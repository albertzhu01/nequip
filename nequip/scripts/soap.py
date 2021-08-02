import torch
import numpy as np
from sklearn import mixture
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from ase.visualize import view
from sklearn.metrics import mean_absolute_error, pairwise
from scipy import stats

from nequip.utils import Config, dataset_from_config
from nequip.data import AtomicDataDict, AtomicData, Collater
from nequip.nn import SequentialGraphNetwork, SaveForOutput

from dscribe.descriptors import SOAP
species = ["H", "C", "O", "N"]
rcut = 6.0
nmax = 8
lmax = 6

# Setting up the SOAP descriptor
soap = SOAP(
    species=species,
    periodic=False,
    rcut=rcut,
    nmax=nmax,
    lmax=lmax,
)

f, ax = plt.subplots(figsize=(19, 9.5))
torch.manual_seed(0)

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

# Sample 100 points from the test set
sample_idxs = torch.randperm(len(dataset_test))[:100]
test_sample = [dataset_test.get(idx.item()) for idx in sample_idxs]

# Create ASE.Atoms from test set
bpa_list = []
for atomic_data in test_sample:
    bpa_list.append(atomic_data.to_ase())
# print(bpa_list)

# Create SOAP output for BPA system
soap_bpa = soap.create(bpa_list)
# print(soap_bpa)
print(soap_bpa.shape)
print(soap_bpa[:, 0, :].shape)

# Evaluate model on test data samples and extract atomic positions and features
c_test = Collater.for_dataset(dataset_test, exclude_keys=[])
test_batch = c_test.collate(test_sample)
print("Begin model evaluation on test data...")
test_out = model(AtomicData.to_AtomicDataDict(test_batch))

test_features = test_out[AtomicDataDict.NODE_FEATURES_KEY]
print(f"Atomic features shape: {test_features.shape}")

# Get dimensions of train and test features and number of atoms in molecule
test_sample_len = len(test_sample)
test_tot_atoms, feature_length = test_features.shape
num_atoms = test_tot_atoms // test_sample_len
print(f"Total test atoms per sample: {test_tot_atoms}")
print(f"Number of atoms in molecule: {num_atoms}")

# Next compute the atomic input and feature distances per atom for all 100 molecules in the 2 test samples
for i in range(num_atoms):
    soap_atom_i = torch.tensor(soap_bpa[:, i, :])
    print(f"Atom {i} SOAP shape: {soap_atom_i.shape}")

    soap_dists = pairwise.cosine_distances(soap_atom_i, soap_atom_i).reshape(-1)
    print(f"Atom {i} SOAP distances shape: {soap_dists.shape}")

    feature_dists = torch.cdist(
        test_features[i:test_tot_atoms:num_atoms].view(1, test_sample_len, feature_length),
        test_features[i:test_tot_atoms:num_atoms].view(1, test_sample_len, feature_length),
        p=2
    ).view(-1)

    print(f"Feature distances shape: {feature_dists.shape}")

    # Plot
    plt.figure()
    plt.subplots(figsize=(16, 9))
    plt.scatter(
        x=soap_dists,
        y=feature_dists.detach().numpy(),
    )
    plt.title(f"3BPA Atom Index {i} Feature Distance vs. SOAP Distance (300K Test)")
    plt.xlabel("SOAP Distance")
    plt.ylabel("Feature Distance")
    plt.savefig(f"bpa_atom{i}_soap-dist_300K.png")
