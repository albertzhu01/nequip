# Merge NPZ batch files by epoch and training/validation
import numpy as np

raw_data = []

num_batches_v = 10

for j in range(num_batches_v):
    tmp_data = np.load('/n/home10/axzhu/nequip/hidden_features/feats_v_epoch2000'
                       + '_batch' + str(j + 1) + '.npz')
    raw_data.append(tmp_data['arr_0'])

np.savez('feats_v_epoch2000.npz',
         batch1=raw_data[0],
         batch2=raw_data[1],
         batch3=raw_data[2],
         batch4=raw_data[3],
         batch5=raw_data[4],
         batch6=raw_data[5],
         batch7=raw_data[6],
         batch8=raw_data[7],
         batch9=raw_data[8],
         batch10=raw_data[9],
         )