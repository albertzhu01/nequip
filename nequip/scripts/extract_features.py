import torch
import numpy as np
from sklearn import mixture
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

from nequip.utils import Config, dataset_from_config
from nequip.data import AtomicDataDict, AtomicData, Collater
from nequip.nn import SequentialGraphNetwork, SaveForOutput

# path = "C:/Users/alber/nequip/nequip/scripts/aspirin_50_epochs_new/results/aspirin/example-run"
path = "/n/home10/axzhu/nequip/results/aspirin/example-run"

model = torch.load(path + "/best_model.pth", map_location=torch.device('cpu'))


# Find the sequential graph net (the bulk of the model):
def find_first_of_type(m: torch.nn.Module, kls) -> torch.nn.Module:
    if isinstance(m, kls):
        return m
    else:
        for child in m.children():
            tmp = find_first_of_type(child, kls)
            if tmp is not None:
                return tmp
    return None


sgn = find_first_of_type(model, SequentialGraphNetwork)

# Now insert a SaveForOutput
insert_after = "layer5_convnet"  # change this
sgn.insert_from_parameters(
    after=insert_after,
    name="feature_extractor",
    shared_params=dict(
        field=AtomicDataDict.NODE_FEATURES_KEY,
        out_field="saved"
    ),
    builder=SaveForOutput
)

# Load a config file
config = Config.from_file(path + "/config_final.yaml")
dataset = dataset_from_config(config)

# Load trainer and get training data indexes and set up Collater
trainer = torch.load(path + '/trainer.pth', map_location='cpu')
train_idxs = trainer['train_idcs']
c = Collater.for_dataset(dataset, exclude_keys=[])

# Create list of training data AtomicData objects
data_list = [dataset.get(idx.item()) for idx in train_idxs]

# Evaluate model on batch of training data
# batch = c.collate(data_list)
# out = model(AtomicData.to_AtomicDataDict(batch))
# assert AtomicDataDict.NODE_FEATURES_KEY in out
# features = out[AtomicDataDict.NODE_FEATURES_KEY].detach().numpy()

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
#     plt.subplots(figsize=(19, 9.5))
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
test_idxs = [idx for idx in range(len(dataset)) if idx not in train_idxs]
test_data_list = [dataset.get(idx) for idx in test_idxs]
test_batch = c.collate(test_data_list)
test_out = model(AtomicData.to_AtomicDataDict(test_batch))
energy_list = [atomic_data.total_energy for atomic_data in test_data_list]
actual_energy = torch.cat(energy_list).view(len(test_data_list), -1)
print(f"Actual energy shape: {actual_energy.shape}")
print(f"Predicted energy shape: {test_out[AtomicDataDict.TOTAL_ENERGY_KEY].shape}")

# probs = gmm.predict_proba(features[:21]).transpose()
# f, ax = plt.subplots(figsize=(19, 9.5))
# prob_plot = sns.heatmap(probs)
# plt.savefig("aspirin_GMM_prob_train_minbic.png")
