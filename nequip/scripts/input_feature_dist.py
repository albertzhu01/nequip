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

# Load trainer and get training and test data indexes and set up Collater
trainer = torch.load(path + '/trainer.pth', map_location='cpu')
train_idxs = trainer['train_idcs']
print(f"# of training points: {len(train_idxs)}")
# val_idxs = trainer['val_idcs']
# print(val_idxs)
# test_idxs = [idx for idx in range(len(dataset)) if idx not in train_idxs]

# Sample two sets of 100 points from the test set
sample_idxs = torch.randperm(len(dataset_test))[:200]
sample1 = sample_idxs[:100]
print(sample1)
sample2 = sample_idxs[100:]
print(sample2)
test_sample1 = [dataset_test.get(idx.item()) for idx in sample1]
test_sample2 = [dataset_test.get(idx.item()) for idx in sample2]

# Create list of training and test data AtomicData objects
# train_data_list = [dataset.get(idx.item()) for idx in train_idxs]
# test_data_list = [dataset_test.get(idx) for idx in range(len(dataset_test))]
# print(f"Train dataset length: {len(train_data_list)}")
# print(f"Test dataset length: {len(test_data_list)}")

# Evaluate model on batch of training data and test data
# Train data
# c = Collater.for_dataset(dataset, exclude_keys=[])
# batch = c.collate(train_data_list)
# print("Begin model evaluation on training data...")
# train_out = model(AtomicData.to_AtomicDataDict(batch))
# train_features = train_out[AtomicDataDict.NODE_FEATURES_KEY].detach().numpy()
# train_pred_forces = train_out[AtomicDataDict.FORCE_KEY].detach().numpy()
# train_a_forces = np.array([atomic_data.forces.detach().numpy() for atomic_data in train_data_list])
# train_actual_forces = train_a_forces.reshape(-1, train_a_forces.shape[-1])
# print(f"train_pred_forces shape: {train_pred_forces.shape}")
# print(f"train_actual_forces shape: {train_actual_forces.shape}")
# train_force_maes = []
# for i in range(len(train_pred_forces)):
#     train_force_maes.append(mean_absolute_error(train_pred_forces[i], train_actual_forces[i]))
# train_force_maes = np.array(train_force_maes)

# Test data
c_test = Collater.for_dataset(dataset_test, exclude_keys=[])
test_batch1 = c_test.collate(test_sample1)
test_batch2 = c_test.collate(test_sample2)
print("Begin model evaluation on test data...")
test_out1 = model(AtomicData.to_AtomicDataDict(test_batch1))
test_out2 = model(AtomicData.to_AtomicDataDict(test_batch2))

test_pos1 = test_out1[AtomicDataDict.POSITIONS_KEY]
test_pos2 = test_out2[AtomicDataDict.POSITIONS_KEY]
print(f"Atomic positions shape: {test_pos1.shape}")

test_features1 = test_out1[AtomicDataDict.NODE_FEATURES_KEY]
test_features2 = test_out2[AtomicDataDict.NODE_FEATURES_KEY]
print(f"Atomic positions shape: {test_features1.shape}")

test_tot_atoms, feature_length = test_features1.shape
num_atoms = test_tot_atoms // len(test_sample1)
print(f"Total test atoms per sample: {test_tot_atoms}")
print(f"Number of atoms in molecule: {num_atoms}")
