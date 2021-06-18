import numpy as np
import torch
import nequip
from ase.io import read
from ase.visualize import view
from nequip.data import AtomicData, AtomicDataDict
from nequip.scripts import deploy
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

# --- Load and test deployed aspirin model on aspirin test data file, extract hidden features --- #
atoms = read("C:/Users/alber/nequip/nequip/scripts/aspirin.xyz")
# view(atoms)
device = "cuda" if torch.cuda.is_available() else "cpu"
model_path = "C:/Users/alber/nequip/nequip/scripts/aspirin_50_epochs_new/results/aspirin/example-run/deployed.pth"
model, metadata = deploy.load_deployed_model(model_path=model_path, device="cpu")
r_max = float(metadata[nequip.scripts.deploy.R_MAX_KEY])
data = AtomicData.from_ase(atoms=atoms, r_max=r_max)
out = model(AtomicData.to_AtomicDataDict(data))
# Extract hidden features
hidden_features = out['feature_vectors'].detach().cpu().numpy()

# plt.plot(hidden_features.transpose())
# plt.show()

# --- Compute average distance between inference and validation atomic features --- #
epoch50 = np.load('C:/Users/alber/nequip/nequip/scripts/aspirin_train_50_epochs/training_features/feats_v_epoch50.npz')
all_euc_dists = []
num_atoms = 21
concat_data_list = []
atom_list = ['C1', 'C2', 'C3', 'C4', 'C5', 'C6', 'C7', 'O1', 'O2', 'O3', 'C8', 'C9', 'O4', 'H1', 'H2', 'H3', 'H4',
             'H5', 'H6', 'H7', 'H8']
inf_atom_list = atom_list

for i in range(num_atoms):
    concat_data_list.append(epoch50['batch1'][i:105:num_atoms])
    for j in range(2, 11):
        concat_data_list[i] = np.concatenate((concat_data_list[i], epoch50[f'batch{j}'][i:105:num_atoms]))

for i in range(50):
    euc_dist_i = [[None for _ in range(num_atoms)] for _ in range(num_atoms)]
    for j in range(num_atoms):
        for k in range(num_atoms):
            atom_inf_feat = hidden_features[j][32:64]
            atom_val_feat = concat_data_list[k][i][32:64]
            euc_dist_i[j][k] = np.linalg.norm(atom_inf_feat - atom_val_feat)
    all_euc_dists.append(euc_dist_i)

all_euc_dists_arr = np.array(all_euc_dists)
euc_dists_arr = np.average(all_euc_dists_arr, axis=0)
print(np.amax(euc_dists_arr))
print(np.amin(euc_dists_arr))

# --- Plot heatmap of average distances --- #
mask = np.zeros_like(euc_dists_arr)
df_euc_dists = pd.DataFrame(data=euc_dists_arr, index=inf_atom_list, columns=atom_list)
f, ax = plt.subplots(figsize=(15, 9.5))
sns.heatmap(df_euc_dists, mask=mask, square=True, cmap='YlGnBu')
plt.title('Average Distance between Inference and Validation Features 33-64 Epoch 50')
plt.ylabel('Atom (Inference)')
plt.xlabel('Atom (Validation)')
plt.show()
