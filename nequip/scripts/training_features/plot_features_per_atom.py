import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

atomic_numbers = [6, 6, 6, 6, 6, 6, 6, 8, 8, 8, 6, 6, 8, 1, 1, 1, 1, 1, 1, 1, 1]
data_C1 = np.empty((1, 240))
epoch1_v = np.load('C:/Users/alber/nequip/nequip/scripts/training_features/feats_v_epoch1.npz')
# print(epoch1_v['batch1'][0:105:21])
# print(data_C1)
for key in epoch1_v.files:
    data_C1 = np.concatenate((data_C1, epoch1_v[key][0:105:21]))

data_C1 = data_C1[1:]
print(data_C1.shape)

stack_data = np.array([])
for i in range(len(data_C1)):
    stack_data = np.concatenate((stack_data, data_C1[i]))

print(stack_data.size)

index = np.array([])
for i in range(50):
    index = np.concatenate((index, np.arange(240)))

index = index.tolist()

df_C1 = pd.DataFrame(stack_data, index=index, columns=['Feature Value'])

print(df_C1)

#
# df_C1 = pd.DataFrame(data_C1)
# df_C1['Index'] = range(240)
#
# print(df_C1)

sns.histplot(df_C1, x=df_C1.index, y='Feature Value', binwidth=(1, 0.004), cbar=True)

plt.title('Carbon 1 Feature Vectors Epoch 1')
plt.xlabel('Feature Vector Index')

plt.show()

# plt.hist2d(list[range(240)], data_C1, bins=40)
# plt.show()