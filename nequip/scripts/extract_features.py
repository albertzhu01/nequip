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

# Find the sequential graph net (the bulk of the model):
# def find_first_of_type(m: torch.nn.Module, kls) -> torch.nn.Module:
#     if isinstance(m, kls):
#         return m
#     else:
#         for child in m.children():
#             tmp = find_first_of_type(child, kls)
#             if tmp is not None:
#                 return tmp
#     return None
#
#
# sgn = find_first_of_type(model, SequentialGraphNetwork)

# Now insert a SaveForOutput
# insert_after = "layer5_convnet"  # change this
# sgn.insert_from_parameters(
#     after=insert_after,
#     name="feature_extractor",
#     shared_params=dict(
#         field=AtomicDataDict.NODE_FEATURES_KEY,
#         out_field="saved"
#     ),
#     builder=SaveForOutput
# )

# Load a config file
config = Config.from_file(path + "/config_final.yaml")
dataset = dataset_from_config(config)
config_test = Config.from_file("/n/home10/axzhu/nequip/configs/dataset.yaml")
dataset_test = dataset_from_config(config_test)

# Load trainer and get training and test data indexes and set up Collater
trainer = torch.load(path + '/trainer.pth', map_location='cpu')
train_idxs = trainer['train_idcs']
val_idxs = trainer['val_idcs']
print(f"# of training points: {len(train_idxs)}")
print(f"# of val points: {len(val_idxs)}")
# val_idxs = trainer['val_idcs']
# print(val_idxs)
# test_idxs = [idx for idx in range(len(dataset)) if idx not in train_idxs]

# Create list of training and test data AtomicData objects
train_data_list = [dataset.get(idx.item()) for idx in train_idxs]
train_data_atoms = [atomic_data.to_ase() for atomic_data in train_data_list]
test_data_list = [dataset_test.get(idx) for idx in range(len(dataset_test))
                  if dataset_test.get(idx).to_ase() not in train_data_atoms]
print(f"Train dataset length: {len(train_data_list)}")
print(f"Test dataset length: {len(test_data_list)}")

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

# Plot force MAEs for a certain atom
# plt.plot(test_force_maes[0:test_tot_atoms:num_atoms])
# plt.savefig("aspirin_C1_test_force_maes.png")

# Get indices and features of best 10 and worst 10 test data for a particular atom and plot euc dists
# for atom_idx in range(7):
#     sorted_args = test_force_maes[atom_idx:test_tot_atoms:num_atoms].argsort()
#     best_test_idxs = sorted_args[:10]
#     print(f"best test idxs: {best_test_idxs}")
#     worst_test_idxs = sorted_args[-10:][::-1]
#     print(f"worst test idxsL {worst_test_idxs}")
#     best_test_features = test_features[atom_idx:test_tot_atoms:num_atoms][best_test_idxs]
#     best_test_force_maes = test_force_maes[atom_idx:test_tot_atoms:num_atoms][best_test_idxs]
#     worst_test_features = test_features[atom_idx:test_tot_atoms:num_atoms][worst_test_idxs]
#     worst_test_force_maes = test_force_maes[atom_idx:test_tot_atoms:num_atoms][worst_test_idxs]
#     print(f"Best test features shape: {best_test_features.shape}, and force MAE:")
#     print(best_test_force_maes)
#     print(f"Worst test features shape: {worst_test_features.shape}, and force MAE:")
#     print(worst_test_force_maes)
#
#     # Calculate Euclidean distance between 10 best and 10 worst features
#     best_worst_features = np.concatenate((best_test_features, worst_test_features))
#     euc_dists = [[None for _ in range(20)] for _ in range(20)]
#     labels = ['Best' for _ in range(10)] + ['Worst' for _ in range(10)]
#
#     for i in range(20):
#         for j in range(i, 20):
#             atom_i_feat = best_worst_features[i]
#             atom_j_feat = best_worst_features[j]
#             euc_dists[i][j] = np.linalg.norm(atom_j_feat - atom_i_feat)
#             euc_dists[j][i] = np.linalg.norm(atom_j_feat - atom_i_feat)
#
#     euc_dists_arr = np.array(euc_dists)
#     mask = np.zeros_like(euc_dists_arr)
#     mask[np.triu_indices_from(mask)] = True
#     df_euc_dists = pd.DataFrame(data=euc_dists_arr, index=labels, columns=labels)
#     plt.figure()
#     plt.subplots(figsize=(12, 9))
#     sns.heatmap(df_euc_dists, mask=mask, square=True, cmap='YlGnBu')
#     plt.title(f'Euclidean Distance between Best and Worst 10 Features of Carbon {atom_idx + 1} (Based on Force MAE)')
#     plt.ylabel('Feature')
#     plt.xlabel('Feature')
#     plt.savefig(f"C{atom_idx + 1}_bw_feature_dist.png")

# Train GMM on training features
# n_components = np.arange(1, 28)
# models = [mixture.GaussianMixture(n_components=n, covariance_type='full', random_state=0) for n in n_components]
# bics = [model.fit(train_features).bic(train_features) for model in models]
# print(f"Number of components with min BIC: {bics.index(min(bics))}")
# gmm = mixture.GaussianMixture(n_components=bics.index(min(bics)), covariance_type='full', random_state=0)
# gmm.fit(train_features)
# print(gmm.converged_)

# Make scatterplot of log-prob vs. force MAE for train and test data for one atom
# for i in range(num_atoms):
#
#     # Training data force MAEs and log-prob densities for 1 atom
#     C1_train_force_maes = train_force_maes[i:train_tot_atoms:num_atoms]
#     C1_train_log_probs = gmm.score_samples(train_features[i:train_tot_atoms:num_atoms])
#
#     # 'Not-so-arbitrary' cut-offs for chemical accuracy and uncertainty
#     mae_cutoff = 0.043
#     logprob_cutoff = np.percentile(C1_train_log_probs, 2.5)
#
#     # Bad (above mae_cutoff) testing data force MAEs and log-prob densities for 1 atom
#     C1_test_force_maes = test_force_maes[i:test_tot_atoms:num_atoms]
#     C1_bad_test_maes_idx = np.where(C1_test_force_maes > mae_cutoff)
#     C1_bad_test_maes = C1_test_force_maes[C1_bad_test_maes_idx]
#     C1_test_log_probs = gmm.score_samples(test_features[i:test_tot_atoms:num_atoms])
#     C1_bad_test_logprobs = C1_test_log_probs[C1_bad_test_maes_idx]
#
#     # Good (below mae_cutoff) testing data force MAEs and log-prob densities for 1 atom
#     C1_good_test_mae_idx = np.setdiff1d(np.arange(len(test_data_list)), C1_bad_test_maes_idx)
#     C1_good_test_maes = C1_test_force_maes[C1_good_test_mae_idx]
#     C1_good_test_logprobs = C1_test_log_probs[C1_good_test_mae_idx]
#
#     # r correlation and p-values for train, total test, bad test, and good test data
#     train_r, train_p = stats.pearsonr(C1_train_force_maes, C1_train_log_probs)
#     test_r, test_p = stats.pearsonr(C1_test_force_maes, C1_test_log_probs)
#     # test_bad_r, test_bad_p = stats.pearsonr(C1_bad_test_maes, C1_bad_test_logprobs)
#     test_good_r, test_good_p = stats.pearsonr(C1_good_test_maes, C1_good_test_logprobs)
#
#     # Number of good and bad test data points, number of each below log-prob cutoff
#     num_test_bad_mae = len(C1_bad_test_maes)
#     num_test_good_mae = len(C1_good_test_maes)
#     num_test_bad_logprob = np.where(C1_bad_test_logprobs < logprob_cutoff)[0].size
#     num_below_l_cutoff = np.where(C1_test_log_probs < logprob_cutoff)[0].size
#     num_test_good_logprob = num_below_l_cutoff - num_test_bad_logprob
#
#     # Plot everything
#     plt.figure()
#     plt.subplots(figsize=(16, 9))
#     plt.scatter(
#         x=C1_good_test_maes,
#         y=C1_good_test_logprobs,
#         color='b',
#         label=f'Test 300K good ({num_test_good_logprob}/{num_test_good_mae}): '
#               + f'r = {test_good_r:.3f}, p-value = {test_good_p:.3f}'
#     )
#     plt.scatter(
#         x=C1_bad_test_maes,
#         y=C1_bad_test_logprobs,
#         color='r',
#         label=f'Test 300K bad ({num_test_bad_logprob}/{num_test_bad_mae})'
#     )
#     plt.scatter(
#         x=C1_train_force_maes,
#         y=C1_train_log_probs,
#         color='k',
#         label=f'Train 300K: r = {train_r:.3f}, p-value: {train_p:.3f}'
#     )
#     plt.axhline(
#         logprob_cutoff,
#         color='k',
#         linestyle='--',
#         label='Uncertainty cutoff (2.5th percentile of training data)'
#     )
#     plt.axvline(mae_cutoff, color='m', linestyle='--', label='Chemical accuracy cutoff')
#     plt.plot([], [], ' ', label=f"All test data: r = {test_r:.3f}, p-value = {test_p:.3f}")
#     plt.legend()
#     plt.title(f"3BPA Atom Index {i} Log-Probability Density vs. Force MAE (300K Test)")
#     plt.xlabel("Force MAE (eV/A)")
#     plt.ylabel("Log-Probability Density")
#     plt.savefig(f"bpa_atom{i}_logprob_vs_mae_300K.png")

# Score samples on training, best 10, and worst 10 features for a particular atom and plot log probs
# C1_train_log_probs = gmm.score_samples(train_features[0:train_tot_atoms:num_atoms])
# C1_best10_log_probs = gmm.score_samples(best_test_features)
# C1_worst10_log_probs = gmm.score_samples(worst_test_features)
# print(f"Training features log-probs shape: {C1_train_log_probs.shape}")
# print(f"Best 10 features log-probs shape: {C1_best10_log_probs.shape}")
# print(C1_best10_log_probs)
# print(f"Worst 10 features log-probs shape: {C1_worst10_log_probs.shape}")
# print(C1_worst10_log_probs)
# plt.hist(
#     C1_train_log_probs,
#     bins=50,
#     color='k',
#     density=True,
#     label='log-probability density, Training, C1')
# plt.hist(
#     C1_best10_log_probs,
#     bins=50,
#     color='b',
#     density=True,
#     label='log-probability density, Best 10, C1'
# )
# plt.hist(
#     C1_worst10_log_probs,
#     bins=50,
#     color='r',
#     density=True,
#     label='log-probability density, Worst 10, C1'
# )
# plt.axvline(
#     np.percentile(C1_train_log_probs, .02),
#     color='c',
#     linestyle='--',
#     label='0.02-th percentile of training configs'
# )
# plt.axvline(
#     np.percentile(C1_train_log_probs, .25),
#     color='g',
#     linestyle='--',
#     label='0.25-th percentile of training configs'
# )
# plt.legend()
# plt.title("Carbon 1 Log-Probability Densities")
# plt.xlabel("Log-Probability Density")
# plt.ylabel("Density")
# plt.savefig("C1_correct_log_probs.png")

# plt.plot(force_maes)
# plt.title("Atomic Force MAE Values for 100 Training Points")
# plt.xlabel("Atom Index")
# plt.ylabel("Atomic Force MAE")
# plt.savefig("aspirin_train_force_maes.png")
# mol_maes = np.sum(force_maes.reshape(100, 21), axis=1)
# print(mol_maes.shape)
# worst_train = np.argmax(mol_maes)
# print(worst_train)

# Plot features
# tot_atoms, feature_length = features.shape
# num_atoms = tot_atoms // len(data_list)
# aspirin_atoms = [
#         "C1",
#         "C2",
#         "C3",
#         "C4",
#         "C5",
#         "C6",
#         "C7",
#         "O1",
#         "O2",
#         "O3",
#         "C8",
#         "C9",
#         "O4",
#         "H1",
#         "H2",
#         "H3",
#         "H4",
#         "H5",
#         "H6",
#         "H7",
#         "H8",
#     ]
# for atom in range(num_atoms):
#     atom_features = features[atom:tot_atoms:num_atoms]
#     atom_features = atom_features.flatten()
#     index = np.tile(np.arange(feature_length), len(data_list))
#     df_atom_features = pd.DataFrame(atom_features, index=index, columns=["Feature Value"])
#     feature_plot = sns.histplot(
#         df_atom_features,
#         x=df_atom_features.index,
#         y=df_atom_features.columns[0],
#         bins=(np.arange(-0.5, 15.6, 1), np.arange(-0.45, 0.45, 0.05)),
#         cbar=True,
#         vmin=0,
#         vmax=100,
#         cmap="viridis",
#     )
#     feature_plot.set(xticks=np.arange(16), ylim=(-0.45, 0.45), yticks=np.arange(-0.45, 0.45, 0.05))
#
#     plt.title(
#         f"{aspirin_atoms[atom]} Training Features Epoch 1220 (Aspirin)"
#     )
#     plt.xlabel("Feature Index")
#     plt.savefig(
#         f"{aspirin_atoms[atom]}_aspirin_features"
#     )

# Train GMM on training features
# Determine optimal number of components using Bayesian inference criterion (BIC)
# n_components = np.arange(1, 20)
# models = [mixture.GaussianMixture(n_components=n, covariance_type='full', random_state=0) for n in n_components]
# aics = [model.fit(features).aic(features) for model in models]
# bics = [model.fit(features).bic(features) for model in models]
# plt.plot(n_components, aics, label='AIC')
# plt.plot(n_components, bics, label='BIC')
# plt.savefig("aspirin_GMM_aics_bics.png")

# Train GMM using optimal number of components
# gmm = mixture.GaussianMixture(n_components=bics.index(min(bics)), covariance_type='full', random_state=0)
# gmm.fit(features)
# print(gmm.converged_)

# Evaluate model on test data
# test_idxs = [idx for idx in range(len(dataset)) if idx not in train_idxs]
# test_data_list = [dataset.get(idx) for idx in test_idxs]
# test_batch = c.collate(test_data_list)
# test_out = model(AtomicData.to_AtomicDataDict(test_batch))
# pred_energy = test_out[AtomicDataDict.TOTAL_ENERGY_KEY]
# energy_list = [atomic_data.total_energy for atomic_data in test_data_list]
# actual_energy = torch.cat(energy_list).view(len(test_data_list), -1)
# print(f"Actual energy shape: {actual_energy.shape}")
# print(f"Predicted energy shape: {pred_energy.shape}")
#
# energy_diff = np.absolute(np.subtract(actual_energy.detach().numpy(), pred_energy.detach().numpy()))
# print(energy_diff.shape)
# max_diff_idx = np.argmax(energy_diff)
# print(max_diff_idx)
# max_diff_test_idx = test_idxs[max_diff_idx]
# print(max_diff_test_idx)    # 52
# plt.plot(energy_diff.flatten())
# plt.savefig("energy_deviations.png")

# Get probability of worst test data point
# worst_test = dataset.get(52)
# out_worst = model(AtomicData.to_AtomicDataDict(worst_test))
# log_probs = gmm.score_samples(features[worst_train * 21: worst_train * 21 + 21, :])
# print(log_probs)
# probs = np.exp(log_probs)
# print(probs)
# log_like = gmm.score(features[worst_train * 21: worst_train * 21 + 21, :])
# print(log_like)
# prob_plot = plt.scatter(np.arange(21), probs)
# plt.title("Weighted Probability Per Atom (Aspirin Worst Training Point)")
# plt.xlabel("Atomic Index")
# plt.ylabel("Weighted Probability")
# plt.savefig("aspirin_score_worst_train.png")

# Get probability of actual worst test data point (and plot the features for C6)
# test_data = np.load(config.dataset_file_name)
# r = test_data['R'][455]
# print(r)
# r[13] = np.array([-2.02, -3.36, 1.44])
# print(r)
# test_atomic_data = AtomicData.from_points(
#     pos=r,
#     r_max=config['r_max'],
#     **{AtomicDataDict.ATOMIC_NUMBERS_KEY:
#         torch.Tensor(torch.from_numpy(test_data['z'].astype(np.float32))).to(torch.int64)}
# )
# view(test_atomic_data.to_ase())
# pred_feature = model(AtomicData.to_AtomicDataDict(test_atomic_data))[AtomicDataDict.NODE_FEATURES_KEY].detach().numpy()
# c6 = pred_feature[5]
# print(c6.shape)
# df_c6 = pd.DataFrame(c6, index=np.arange(len(c6)), columns=['Feature Value'])
# plt.subplots(figsize=(19, 9.5))
# c6_plot = sns.histplot(
#     df_c6,
#     x=df_c6.index,
#     y=df_c6.columns[0],
#     bins=(np.arange(-0.5, 15.6, 1), np.arange(-0.45, 0.45, 0.05)),
#     cbar=True,
#     vmin=0,
#     vmax=100,
#     cmap="viridis"
# )
# c6_plot.set(xticks=np.arange(16), ylim=(-0.45, 0.45), yticks=np.arange(-0.45, 0.45, 0.05))
#
# plt.title(
#     f"C6 Features for Aspirin with 2x O-H Bond Length"
# )
# plt.xlabel("Feature Index")
# plt.savefig(
#     f"C6_2x_O-H_len_featues.png"
# )
# prob = gmm.predict_proba(pred_feature.detach().numpy()).transpose()
# prob_plot = sns.heatmap(prob)
# plt.savefig("aspirin_GMM_2x_O-H_len.png")
