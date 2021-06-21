import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

atomic_numbers = [6, 6, 6, 6, 6, 6, 6, 8, 8, 8, 6, 6, 8, 1, 1, 1, 1, 1, 1, 1, 1]

sns.set(rc={'axes.facecolor': 'ffffff',
            'figure.facecolor': 'white',
            'axes.edgecolor': 'black',
            'axes.grid': False,
            'xtick.bottom': True,
            'ytick.left': True})

# --- Code for creating 2d histograms per epoch. Uncomment one of lines 10-45 OR lines 51-88 --- #
data_C1 = np.empty((1, 240))
epoch1_v = np.load('C:/Users/alber/nequip/nequip/scripts/aspirin_train_50_epochs/training_features/feats_v_epoch50.npz')

# largest = 0
# smallest = 0
# for key in epoch1_v.files:
#     if np.amax(epoch1_v[key]) > largest:
#         largest = np.amax(epoch1_v[key])
#     if np.amin(epoch1_v[key]) < smallest:
#         smallest = np.amin(epoch1_v[key])
#
# print(largest)
# print(smallest)

for key in epoch1_v.files:
    data_C1 = np.concatenate((data_C1, epoch1_v[key][0:105:21]))
    # data_C1 = np.concatenate((data_C1, epoch1_v[key][1:105:21]))
    # data_C1 = np.concatenate((data_C1, epoch1_v[key][2:105:21]))
    # data_C1 = np.concatenate((data_C1, epoch1_v[key][3:105:21]))

data_C1 = data_C1[1:]
print(data_C1.shape)

stack_data = np.array([])
for i in range(len(data_C1)):
    stack_data = np.concatenate((stack_data, data_C1[i]))

rows, cols = data_C1.shape

index = np.array([])
for i in range(rows):
    index = np.concatenate((index, np.arange(cols)))

index = index.tolist()

df_C1 = pd.DataFrame(stack_data, index=index, columns=['Feature Value'])
print(df_C1)

f, ax = plt.subplots(figsize=(22, 9.6))

feature_plot = sns.histplot(df_C1,
                            x=df_C1.index,
                            y='Feature Value',
                            binwidth=(1, 0.03),
                            cbar=True,
                            vmin=0,
                            vmax=50,
                            cmap='viridis'
                            )
feature_plot.set(xticks=list(range(0, 250, 10)),
                 ylim=(-1.5, 1.5),
                 yticks=np.arange(-1.5, 1.6, 0.3).tolist())

plt.title('Carbon 1 Features Epoch 50')
plt.xlabel('Feature Index')

plt.show()


# --- Code for creating line plots of averages --- #

data_C1_all_epochs = []

# - Code for creating line plots for multiple epochs. Uncomment one of lines 52-60 OR lines 63-73 - #
# for i in 10, 50:
#     epoch_data = np.load('C:/Users/alber/nequip/nequip/scripts/aspirin_train_50_epochs/feats_v_epoch' + str(i) + '.npz')
#     tmp_data = np.empty((1, 240))
#     for key in epoch_data.files:
#         tmp_data = np.concatenate((tmp_data, epoch_data[key][20:105:21]))
#
#     tmp_data = tmp_data[1:]
#     avg_data_C1 = np.average(tmp_data, axis=0)
#     data_C1_all_epochs.append(avg_data_C1)

# - Code for creating line plots for one epoch. Uncomment one of lines 52-60 OR lines 63-73 - #
# epoch_data = np.load('C:/Users/alber/nequip/nequip/scripts/aspirin_train_50_epochs/feats_v_epoch50.npz')
#
# tmp_data = np.empty((1, 240))
# for key in epoch_data.files:
#     tmp_data = np.concatenate((tmp_data, epoch_data[key][7:105:21]))
#     tmp_data = np.concatenate((tmp_data, epoch_data[key][8:105:21]))
#     tmp_data = np.concatenate((tmp_data, epoch_data[key][20:105:21]))
#     tmp_data = np.concatenate((tmp_data, epoch_data[key][3:105:21]))
#
# tmp_data = tmp_data[1:]
# avg_data_C1 = np.average(tmp_data, axis=0)
# data_C1_all_epochs.append(avg_data_C1)
#
# # - Create line plot of the dataframe - #
# data_C1_all_epochs = np.array(data_C1_all_epochs)
# print(data_C1_all_epochs.shape)
#
# df_C1_all_epochs = pd.DataFrame(data_C1_all_epochs, index=['Epoch 50'])
# df_C1_all_epochs = df_C1_all_epochs.transpose()
# print(df_C1_all_epochs)

# df_C1_all_epochs.plot(kind='line', alpha=1, ylim=(-0.45, 0.45), color=['darkblue'])
# plt.legend(loc='lower right')
# plt.title('Oxygens 1, 2 Validation Features (Averaged 50 Per Atom)')
# plt.xlabel('Feature Index')
# plt.ylabel('Feature Value')
# plt.show()
