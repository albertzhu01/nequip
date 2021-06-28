# Merge NPZ batch files by epoch and training/validation
import numpy as np

raw_data = []
num_epochs = 50
num_batches_v = 10

for i in range(4, num_epochs, 5):
    for j in range(num_batches_v):
        tmp_data = np.load('C:/Users/alber/nequip/nequip/scripts/aspirin_train_50_epochs/feats_v_epoch'
                           + str(i + 1) + '_batch' + str(j + 1) + '.npz')
        raw_data.append(tmp_data['arr_0'])

    np.savez('feats_v_epoch' + str(i + 1) + '.npz',
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
    raw_data = []
