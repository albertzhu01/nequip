import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import numpy as np
import pandas as pd
import seaborn as sns

atomic_numbers = [6, 6, 6, 6, 6, 6, 6, 8, 8, 8, 6, 6, 8, 1, 1, 1, 1, 1, 1, 1, 1]

# Code for creating 2d histograms per epoch
# data_C1 = np.empty((1, 240))
# epoch1_v = np.load('C:/Users/alber/nequip/nequip/scripts/training_features/feats_v_epoch10.npz')
#
# for key in epoch1_v.files:
#     data_C1 = np.concatenate((data_C1, epoch1_v[key][6:105:21]))
#
# data_C1 = data_C1[1:]
# print(data_C1.shape)
#
# stack_data = np.array([])
# for i in range(len(data_C1)):
#     stack_data = np.concatenate((stack_data, data_C1[i]))
#
# rows, cols = data_C1.shape
#
# index = np.array([])
# for i in range(rows):
#     index = np.concatenate((index, np.arange(cols)))
#
# index = index.tolist()
#
# df_C1 = pd.DataFrame(stack_data, index=index, columns=['Feature Value'])
# print(df_C1)
#
# feature_plot = sns.histplot(df_C1,
#                             x=df_C1.index,
#                             y='Feature Value',
#                             binwidth=(1, 0.01),
#                             cbar=True
#                             )
# feature_plot.set(ylim=(-1, 1))
#
# plt.title('Carbon 7 Features Epoch 10')
# plt.xlabel('Feature Index')
#
# plt.show()

# Code for creating line plots
data_C1_all_epochs = []
# for i in 1, 5, 10:
#     epoch_data = np.load('C:/Users/alber/nequip/nequip/scripts/training_features/feats_v_epoch' + str(i) + '.npz')
#     tmp_data = np.empty((1, 240))
#     for key in epoch_data.files:
#         tmp_data = np.concatenate((tmp_data, epoch_data[key][0:105:21]))
#
#     tmp_data = tmp_data[1:]
#     avg_data_C1 = np.average(tmp_data, axis=0)
#     data_C1_all_epochs.append(avg_data_C1)

epoch_data = np.load('C:/Users/alber/nequip/nequip/scripts/training_features/feats_v_epoch10.npz')
tmp_data = np.empty((1, 240))
for key in epoch_data.files:
    tmp_data = np.concatenate((tmp_data, epoch_data[key][14:105:21]))
    tmp_data = np.concatenate((tmp_data, epoch_data[key][15:105:21]))
    tmp_data = np.concatenate((tmp_data, epoch_data[key][16:105:21]))
    tmp_data = np.concatenate((tmp_data, epoch_data[key][17:105:21]))

tmp_data = tmp_data[1:]
avg_data_C1 = np.average(tmp_data, axis=0)
data_C1_all_epochs.append(avg_data_C1)

data_C1_all_epochs = np.array(data_C1_all_epochs)
print(data_C1_all_epochs.shape)

df_C1_all_epochs = pd.DataFrame(data_C1_all_epochs, index=['Epoch 10'])
df_C1_all_epochs = df_C1_all_epochs.transpose()
print(df_C1_all_epochs)

df_C1_all_epochs.plot(kind='line', alpha=0.6, ylim=(-0.45, 0.45), color='blue')
plt.legend(loc='lower right')
plt.title('Hydrogens 2, 3, 4, 5 Validation Features (Averaged 50 Per Atom)')
plt.xlabel('Feature Index')
plt.ylabel('Feature Value')
plt.show()