import torch
import numpy as np
from sklearn import mixture
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from ase.visualize import view
from sklearn.metrics import mean_absolute_error
from sklearn.linear_model import LogisticRegression
from scipy import stats

from nequip.utils import Config, dataset_from_config
from nequip.data import AtomicDataDict, AtomicData, Collater
from nequip.nn import SequentialGraphNetwork, SaveForOutput

f, ax = plt.subplots(figsize=(19, 9.5))

# path = "C:/Users/alber/nequip/nequip/scripts/aspirin_50_epochs_new/results/aspirin/example-run"
path = "/n/home10/axzhu/nequip/results/aspirin/example-run"

model = torch.load(path + "/best_model.pth", map_location=torch.device('cpu'))
model.eval()

# Load a config file
config = Config.from_file(path + "/config_final.yaml")
dataset = dataset_from_config(config)
# print(len(dataset))

# Load trainer and get training and test data indexes and set up Collater
trainer = torch.load(path + '/trainer.pth', map_location='cpu')
train_idxs = trainer['train_idcs']
val_idxs = trainer['val_idcs']
test_idxs = [idx for idx in range(len(dataset)) if idx not in torch.cat((train_idxs, val_idxs))]
c = Collater.for_dataset(dataset, exclude_keys=[])

# Create list of training, validation, and test data AtomicData objects
train_data_list = [dataset.get(idx.item()) for idx in train_idxs]
val_data_list = [dataset.get(idx.item()) for idx in val_idxs]
test_data_list = [dataset.get(idx) for idx in test_idxs]

# Evaluate model on batch of training data, val data, and test data
# Train data
batch = c.collate(train_data_list)
train_out = model(AtomicData.to_AtomicDataDict(batch))
train_features = train_out[AtomicDataDict.NODE_FEATURES_KEY].detach().numpy()
train_pred_forces = train_out[AtomicDataDict.FORCE_KEY].detach().numpy()
train_a_forces = np.array([atomic_data.forces.detach().numpy() for atomic_data in train_data_list])
train_actual_forces = train_a_forces.reshape(-1, train_a_forces.shape[-1])
print(f"train_pred_forces shape: {train_pred_forces.shape}")
print(f"train_actual_forces shape: {train_actual_forces.shape}")
train_force_maes = []
for i in range(len(train_pred_forces)):
    train_force_maes.append(mean_absolute_error(train_pred_forces[i], train_actual_forces[i]))
train_force_maes = np.array(train_force_maes)
train_bad_label = np.where(train_force_maes > 1.5, np.ones(train_force_maes.size), np.zeros(train_force_maes.size))

# Val data
val_batch = c.collate(val_data_list)
val_out = model(AtomicData.to_AtomicDataDict(val_batch))
val_features = val_out[AtomicDataDict.NODE_FEATURES_KEY].detach().numpy()
val_pred_forces = val_out[AtomicDataDict.FORCE_KEY].detach().numpy()
val_a_forces = np.array([atomic_data.forces.detach().numpy() for atomic_data in val_data_list])
val_actual_forces = val_a_forces.reshape(-1, val_a_forces.shape[-1])
print(f"train_pred_forces shape: {val_pred_forces.shape}")
print(f"train_actual_forces shape: {val_actual_forces.shape}")
val_force_maes = []
for i in range(len(val_pred_forces)):
    val_force_maes.append(mean_absolute_error(val_pred_forces[i], val_actual_forces[i]))
val_force_maes = np.array(val_force_maes)
val_bad_label = np.where(val_force_maes > 1.5, np.ones(val_force_maes.size), np.zeros(val_force_maes.size))

# Test data
test_batch = c.collate(test_data_list)
test_out = model(AtomicData.to_AtomicDataDict(test_batch))
test_features = test_out[AtomicDataDict.NODE_FEATURES_KEY].detach().numpy()
test_pred_forces = test_out[AtomicDataDict.FORCE_KEY].detach().numpy()
test_a_forces = np.array([atomic_data.forces.detach().numpy() for atomic_data in test_data_list])
test_actual_forces = test_a_forces.reshape(-1, train_a_forces.shape[-1])
print(f"test_pred_forces shape: {test_pred_forces.shape}")
print(f"test_actual_forces shape: {test_actual_forces.shape}")
test_force_maes = []
for i in range(len(test_pred_forces)):
    test_force_maes.append(mean_absolute_error(test_pred_forces[i], test_actual_forces[i]))
test_force_maes = np.array(test_force_maes)
test_bad_label = np.where(test_force_maes > 1.5, np.ones(test_force_maes.size), np.zeros(test_force_maes.size))

# Get dimensions of train, val, and test features and number of atoms in aspirin
train_tot_atoms, feature_length = train_features.shape
num_atoms = train_tot_atoms // len(train_data_list)
test_tot_atoms, _ = test_features.shape
val_tot_atoms, _ = val_features.shape
print(f"num_atoms: {num_atoms}")
print(f"total train atoms: {train_tot_atoms}")
print(f"total test atoms: {test_tot_atoms}")
print(f"total val atoms: {val_tot_atoms}")

feature_clf = LogisticRegression(
    class_weight='balanced',
    solver='liblinear',
    random_state=0,
    max_iter=1000
).fit(val_features[0:val_tot_atoms:num_atoms], val_bad_label)

train_accuracy = feature_clf.score(
    train_features[0:train_tot_atoms:num_atoms],
    train_bad_label[0:train_tot_atoms:num_atoms]
)
print(f"Train dataset accuracy: {train_accuracy}")

val_accuracy = feature_clf.score(
    val_features[0:val_tot_atoms:num_atoms],
    val_bad_label[0:val_tot_atoms:num_atoms]
)
print(f"Val dataset accuracy: {val_accuracy}")

test_accuracy = feature_clf.score(
    test_features[0:test_tot_atoms:num_atoms],
    test_bad_label[0:test_tot_atoms:num_atoms]
)
print(f"Test dataset accuracy: {test_accuracy}")
