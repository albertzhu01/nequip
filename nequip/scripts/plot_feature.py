import matplotlib.pyplot as plt
import numpy as np

data1 = np.load('C:/Users/alber/nequip/nequip/scripts/features_train_batch1.npz')
data2 = np.load('C:/Users/alber/nequip/nequip/scripts/features_train_batch20.npz')
val1 = np.load('C:/Users/alber/nequip/nequip/scripts/features_val_batch1.npz')
val2 = np.load('C:/Users/alber/nequip/nequip/scripts/features_val_batch10.npz')
train_features = np.concatenate((data1['arr_0'], data2['arr_0']), axis=0)
for i in range(len(train_features) // 2):
    plt.plot(train_features[i])
plt.show()

for i in range(len(train_features) // 2, len(train_features)):
    plt.plot(train_features[i])
plt.show()

# val_features = np.concatenate((val1['arr_0'], val2['arr_0']), axis=0)
# for i in range(len(val_features) // 2):
#     plt.plot(val_features[i])
# plt.show()
#
# for i in range(len(val_features) // 2, len(val_features)):
#     plt.plot(val_features[i])
# plt.show()

