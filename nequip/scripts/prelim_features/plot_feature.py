import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

atomic_numbers = [6, 6, 6, 6, 6, 6, 6, 8, 8, 8, 6, 6, 8, 1, 1, 1, 1, 1, 1, 1, 1]
atoms = []
for j in range(5):
    for i in range(len(atomic_numbers)):
        if atomic_numbers[i] == 6:
            atoms.append('Carbon')
        elif atomic_numbers[i] == 8:
            atoms.append('Oxygen')
        else:
            atoms.append('Hydrogen')

data1 = np.load('C:/Users/alber/nequip/nequip/scripts/prelim_features/features_train_batch1.npz')
data2 = np.load('C:/Users/alber/nequip/nequip/scripts/prelim_features/features_train_batch20.npz')
val1 = np.load('C:/Users/alber/nequip/nequip/scripts/prelim_features/features_val_batch1.npz')
val2 = np.load('C:/Users/alber/nequip/nequip/scripts/prelim_features/features_val_batch10.npz')

df_trainbatch20 = pd.DataFrame(data=data2['arr_0'][0:104:21], index=None, columns=None)
df_valbatch10 = pd.DataFrame(data=val2['arr_0'][0:21:7], index=['C', 'O', 'H'], columns=None)
# df_valbatch10['Atom'] = ['C', 'O', 'H']
df_valbatch10 = df_valbatch10.transpose()
# df_trainbatch20['Atom'] = atoms

print(df_valbatch10)

# plt.figure()
#
# pd.plotting.parallel_coordinates(
#                                  frame=df_valbatch10,
#                                  class_column='Atom',
#                                  color=('#556270', '#4ECDC4', '#C7F464'),
#                                  axvlines=False,
#                                  xticks=list(range(240))
#                                 )

df_valbatch10.plot.line()

plt.show()

# plt.figure()
#
# pd.plotting.parallel_coordinates(
#                                  frame=df_trainbatch20,
#                                  class_column='Atom',
#                                  color=('#556270', '#4ECDC4', '#C7F464'),
#                                  axvlines=False,
#                                  xticks=list(range(240))
#                                 )
#
# plt.show()

#
# df_trainbatch20.plot.line(colormap='winter')
#
# plt.title('Feature Vectors for Final Training Batch (1 Epoch)')
# plt.xlabel('Feature Index')
# plt.ylabel('Value')
#
# plt.show()


# train_features = np.concatenate((data1['arr_0'], data2['arr_0']), axis=0)
# for i in range(len(train_features) // 2):
#     plt.plot(train_features[i])
# plt.show()
#
# for i in range(len(train_features) // 2, len(train_features)):
#     plt.plot(train_features[i])
# plt.show()
#
# val_features = np.concatenate((val1['arr_0'], val2['arr_0']), axis=0)
# for i in range(len(val_features) // 2):
#     plt.plot(val_features[i])
# plt.show()
#
# for i in range(len(val_features) // 2, len(val_features)):
#     plt.plot(val_features[i])
# plt.show()




# train_data = np.load('C:/Users/alber/nequip/nequip/scripts/benchmark_data/aspirin_ccsd-train.npz')
# for i in train_data.files:
#     print(i)
#     print(train_data[i])