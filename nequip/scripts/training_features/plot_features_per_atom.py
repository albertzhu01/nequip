import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

atomic_numbers = [6, 6, 6, 6, 6, 6, 6, 8, 8, 8, 6, 6, 8, 1, 1, 1, 1, 1, 1, 1, 1]

# Code for creating 2d histograms per epoch
# data_C1 = np.empty((1, 240))
# epoch1_v = np.load('C:/Users/alber/nequip/nequip/scripts/training_features/feats_v_epoch10.npz')
#
# for key in epoch1_v.files:
#     data_C1 = np.concatenate((data_C1, epoch1_v[key][0:105:21]))
#
# data_C1 = data_C1[1:]
# print(data_C1.shape)
#
# stack_data = np.array([])
# for i in range(len(data_C1)):
#     stack_data = np.concatenate((stack_data, data_C1[i]))
#
# print(stack_data.size)
#
# index = np.array([])
# for i in range(50):
#     index = np.concatenate((index, np.arange(240)))
#
# index = index.tolist()
#
# df_C1 = pd.DataFrame(stack_data, index=index, columns=['Feature Value'])
#
# sns.histplot(df_C1, x=df_C1.index, y='Feature Value', binwidth=(1, 0.005), cbar=True)
#
# plt.title('Carbon 1 Feature Vectors Epoch 10')
# plt.xlabel('Feature Vector Index')
#
# plt.show()

# Code for creating line plots
data_C1_all_epochs = []
for i in 1, 5, 10:
    epoch_data = np.load('C:/Users/alber/nequip/nequip/scripts/training_features/feats_v_epoch' + str(i) + '.npz')
    tmp_data = np.empty((1, 240))
    for key in epoch_data.files:
        tmp_data = np.concatenate((tmp_data, epoch_data[key][11:105:21]))

    tmp_data = tmp_data[1:]
    avg_data_C1 = np.average(tmp_data, axis=0)
    data_C1_all_epochs.append(avg_data_C1)

data_C1_all_epochs = np.array(data_C1_all_epochs)
print(data_C1_all_epochs.shape)

df_C1_all_epochs = pd.DataFrame(data_C1_all_epochs, index=['Epoch 1', 'Epoch 5', 'Epoch 10'])
df_C1_all_epochs = df_C1_all_epochs.transpose()
print(df_C1_all_epochs)

df_C1_all_epochs.plot(kind='line', alpha=0.6, ylim=(-0.45, 0.45))
plt.legend(loc='lower right')
plt.title('Carbon9 Training Validation Features (50-Atom Average)')
plt.xlabel('Feature Index')
plt.ylabel('Feature Value')
plt.show()