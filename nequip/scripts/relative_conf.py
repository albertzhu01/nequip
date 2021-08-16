import torch
import numpy as np
from sklearn import mixture
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error

from nequip.utils import Config, dataset_from_config
from nequip.data import AtomicDataDict, AtomicData, Collater

# path = "C:/Users/alber/nequip/nequip/scripts/aspirin_50_epochs_new/results/aspirin/example-run"
path_300K = "/n/home10/axzhu/nequip/results/bpa/train300K_072321"
path_mixed = "/n/home10/axzhu/nequip/results/bpa_mixed/train_mixed_072721"

model_300K = torch.load(path_300K + "/best_model.pth", map_location=torch.device('cpu'))
model_mixed = torch.load(path_mixed + "/best_model.pth", map_location=torch.device('cpu'))
model_300K.eval()
model_mixed.eval()

# Load config files
config_300K_train = Config.from_file(path_300K + "/config_final.yaml")
dataset_300K_train = dataset_from_config(config_300K_train)

config_mixed_train = Config.from_file(path_mixed + "/config_final.yaml")
dataset_mixed_train = dataset_from_config(config_mixed_train)

config_300K_test = Config.from_file("/n/home10/axzhu/nequip/configs/bpa_300K.yaml")
dataset_300K_test = dataset_from_config(config_300K_test)

config_600K_test = Config.from_file("/n/home10/axzhu/nequip/configs/bpa_600K.yaml")
dataset_600K_test = dataset_from_config(config_600K_test)

config_1200K_test = Config.from_file("/n/home10/axzhu/nequip/configs/bpa_1200K.yaml")
dataset_1200K_test = dataset_from_config(config_1200K_test)

# Load trainers and get training and test data indexes and set up Collater
trainer_300K = torch.load(path_300K + '/trainer.pth', map_location='cpu')
train_idxs_300K = trainer_300K['train_idcs']
val_idxs_300K = trainer_300K['val_idcs']

trainer_mixed = torch.load(path_mixed + '/trainer.pth', map_location='cpu')
train_idxs_mixed = trainer_mixed['train_idcs']
val_idxs_mixed = trainer_mixed['val_idcs']
# print(f"# of training points: {len(train_idxs)}")
# print(f"# of val points: {len(val_idxs)}")
# val_idxs = trainer['val_idcs']
# print(val_idxs)
# test_idxs = [idx for idx in range(len(dataset)) if idx not in train_idxs]


# Create list of training and test data AtomicData objects
def get_test_idxs(train_dataset, train_idxs, val_idxs, test_dataset):
    train_data_list = [train_dataset.get(idx.item()) for idx in train_idxs]
    train_val_data_list = [train_dataset.get(idx.item()) for idx in torch.cat((train_idxs, val_idxs))]
    train_data_atoms = [atomic_data.to_ase() for atomic_data in train_val_data_list]
    test_data_list = [test_dataset.get(idx) for idx in range(len(test_dataset))
                      if test_dataset.get(idx).to_ase() not in train_data_atoms]
    return train_data_list, test_data_list


train_300K, tr_300K_test_300K = get_test_idxs(dataset_300K_train, train_idxs_300K, val_idxs_300K, dataset_300K_test)
_, tr_300K_test_600K = get_test_idxs(dataset_300K_train, train_idxs_300K, val_idxs_300K, dataset_600K_test)
_, tr_300K_test_1200K = get_test_idxs(dataset_300K_train, train_idxs_300K, val_idxs_300K, dataset_1200K_test)
train_mix, tr_mix_test_300K = get_test_idxs(dataset_mixed_train, train_idxs_300K, val_idxs_300K, dataset_300K_test)
_, tr_mix_test_600K = get_test_idxs(dataset_mixed_train, train_idxs_300K, val_idxs_300K, dataset_600K_test)
_, tr_mix_test_1200K = get_test_idxs(dataset_mixed_train, train_idxs_300K, val_idxs_300K, dataset_1200K_test)

# print(f"Train + val dataset length: {len(train_val_data_list)}")
# print(f"Test dataset length: {len(test_data_list)}")
# print(f"Test idxs length: {len(test_idxs)}")
# print(test_idxs)

# Evaluate model on batch of training data and test data
# Train data
# c_train_300K = Collater.for_dataset(dataset_300K_train, exclude_keys=[])
# batch_train_300K = c_train_300K.collate(train_300K)
# print("Begin model evaluation on 300K training data...")
# train_out = model_300K(AtomicData.to_AtomicDataDict(batch_train_300K))
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
# c_test = Collater.for_dataset(dataset_test, exclude_keys=[])
# test_batch = c_test.collate(test_data_list)
# print("Begin model evaluation on test data...")
# test_out = model(AtomicData.to_AtomicDataDict(test_batch))
# test_features = test_out[AtomicDataDict.NODE_FEATURES_KEY].detach().numpy()
# test_pred_forces = test_out[AtomicDataDict.FORCE_KEY].detach().numpy()
# test_a_forces = np.array([atomic_data.forces.detach().numpy() for atomic_data in test_data_list])
# test_actual_forces = test_a_forces.reshape(-1, train_a_forces.shape[-1])
# print(f"test_pred_forces shape: {test_pred_forces.shape}")
# print(f"test_actual_forces shape: {test_actual_forces.shape}")
# test_force_maes = []
# for i in range(len(test_pred_forces)):
#     test_force_maes.append(mean_absolute_error(test_pred_forces[i], test_actual_forces[i]))
# test_force_maes = np.array(test_force_maes)

# Get dimensions of train and test features and number of atoms in aspirin
# train_tot_atoms, feature_length = train_features.shape
# num_atoms = train_tot_atoms // len(train_data_list)
# test_tot_atoms, _ = test_features.shape
# print(f"num_atoms: {num_atoms}")
# print(f"total train atoms: {train_tot_atoms}")
# print(f"total test atoms: {test_tot_atoms}")

# Train GMM on training features
# n_components = np.arange(1, 28)
# models = [mixture.GaussianMixture(n_components=n, covariance_type='full', random_state=0) for n in n_components]
# bics = [model.fit(train_features).bic(train_features) for model in models]
# print(f"Number of components with min BIC: {bics.index(min(bics))}")
# gmm = mixture.GaussianMixture(n_components=bics.index(min(bics)), covariance_type='full', random_state=0)
# gmm.fit(train_features)
# print(gmm.converged_)

# Plot force MAE and RMSE vs. relative confidence for each atom
# percentiles = np.arange(0, 100, 5)
# for i in range(num_atoms):
#     atom_log_probs = gmm.score_samples(test_features[i:test_tot_atoms:num_atoms])
#     # print(f"atom_log_probs shape: {atom_log_probs.shape}")
#     atom_force_maes = test_force_maes[i:test_tot_atoms:num_atoms]
#     atom_pred_forces = test_pred_forces[i:test_tot_atoms:num_atoms]
#     # print(f"atom_pred_forces shape: {atom_pred_forces.shape}")
#     atom_actual_forces = test_actual_forces[i:test_tot_atoms:num_atoms]
#     # print(f"atom_actual_forces shape: {atom_actual_forces.shape}")
#     f_maes = []
#     f_rmses = []
#
#     for j in np.nditer(percentiles):
#         # print(f"percentile: {np.percentile(atom_log_probs, j)}")
#         cutoff_idxs = np.argwhere(atom_log_probs >= np.percentile(atom_log_probs, j)).reshape(-1)
#         # print(f"cutoff_idxs shape: {cutoff_idxs.shape}")
#         f_maes.append(np.mean(atom_force_maes[cutoff_idxs]))
#         # print(f"atom_pred_forces[cutoff_idxs] shape: {atom_pred_forces[cutoff_idxs].shape}")
#         # print(f"atom_actual_forces[cutoff_idxs] shape: {atom_actual_forces[cutoff_idxs].shape}")
#         # print(f"f_rmses: {f_rmses}")
#         f_rmses.append(
#             mean_squared_error(atom_actual_forces[cutoff_idxs], atom_pred_forces[cutoff_idxs], squared=False)
#         )
#
#     plt.figure()
#     plt.subplots(figsize=(16, 9))
#     plt.plot(
#         percentiles,
#         f_maes,
#         color='b',
#         marker='o',
#         label=f'Force MAE'
#     )
#     plt.plot(
#         percentiles,
#         f_rmses,
#         color='g',
#         marker='o',
#         label=f'Force RMSE'
#     )
#     plt.legend(fontsize=14)
#     plt.title(f"3BPA Atom Index {i} Error vs. Relative Confidence (300K Train, 300K Test)", fontsize=18)
#     plt.xlabel("Confidence Percentile", fontsize=16)
#     plt.ylabel("Error (eV/A)", fontsize=16)
#     plt.savefig(f"bpa_atom{i}_err_vs_rel-conf_300K.png")
