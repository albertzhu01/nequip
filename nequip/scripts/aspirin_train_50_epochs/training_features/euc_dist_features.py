import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

#--- Distance between atomic features at Epoch 50 --- #

atom_list = ['C1', 'C2', 'C3', 'C4', 'C5', 'C6', 'C7', 'O1', 'O2', 'O3', 'C8', 'C9', 'O4', 'H1', 'H2', 'H3', 'H4',
             'H5', 'H6', 'H7', 'H8']

epoch50 = np.load('C:/Users/alber/nequip/nequip/scripts/aspirin_train_50_epochs/training_features/feats_v_epoch50.npz')

concat_data_list = []
for i in range(21):
    concat_data_list.append(epoch50['batch1'][i:105:21])
    for j in range(2, 11):
        concat_data_list[i] = np.concatenate((concat_data_list[i], epoch50['batch' + str(j)][i:105:21]))
    concat_data_list[i] = np.average(concat_data_list[i], axis=0)

for i in range(21):
    print(concat_data_list[i].shape)

num_atoms = len(concat_data_list)
euc_dists = [[None for _ in range(21)] for _ in range(21)]

for i in range(num_atoms):
    for j in range(i, num_atoms):
        atom_i_feat = concat_data_list[i][200:240:1]
        atom_j_feat = concat_data_list[j][200:240:1]
        euc_dists[i][j] = np.linalg.norm(atom_j_feat - atom_i_feat)
        euc_dists[j][i] = np.linalg.norm(atom_j_feat - atom_i_feat)

for i in range(num_atoms):
    for j in range(i + 1, num_atoms):
        print(euc_dists[i][j], end=' ')
    print()

euc_dists_arr = np.array(euc_dists)
mask = np.zeros_like(euc_dists_arr)
mask[np.triu_indices_from(mask)] = True
df_euc_dists = pd.DataFrame(data=euc_dists_arr, index=atom_list, columns=atom_list)
# with sns.axes_style("white"):
f, ax = plt.subplots(figsize=(11, 9.5))
sns.heatmap(df_euc_dists, mask=mask, square=True, cmap='YlGnBu')
plt.title('Euclidean Distance between Atomic Feature Indices 201-240 Epoch 50')
plt.ylabel('Atom')
plt.xlabel('Atom')
plt.show()
