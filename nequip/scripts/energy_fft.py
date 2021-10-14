import torch
import numpy as np
from sklearn import mixture
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from ase.visualize import view
from sklearn.metrics import mean_absolute_error, mean_squared_error
from scipy import stats
from scipy.fftpack import fft, fftfreq

from nequip.utils import Config, dataset_from_config
from nequip.data import AtomicDataDict, AtomicData, Collater
from nequip.nn import SequentialGraphNetwork, SaveForOutput

f, ax = plt.subplots(figsize=(16, 9))

# path = "C:/Users/alber/nequip/nequip/scripts/aspirin_50_epochs_new/results/aspirin/example-run"
path = "/n/home10/axzhu/nequip/results/aspirin-md17/example-run-full"

model = torch.load(path + "/best_model.pth", map_location=torch.device('cpu'))
model.eval()

# Load a config file
config = Config.from_file(path + "/config_final.yaml")
dataset = dataset_from_config(config)

# Load trainer and get training and test data indexes and set up Collater
trainer = torch.load(path + '/trainer.pth', map_location='cpu')
train_idxs = trainer['train_idcs']
val_idxs = trainer['val_idcs']
print(f"# of training points: {len(train_idxs)}")
print(f"# of val points: {len(val_idxs)}")

# Create list of training and test data AtomicData objects
N = 10000
test_idxs = [idx for idx in range(N + 1000) if idx not in torch.cat((train_idxs, val_idxs)).tolist()]
test_data_list = [dataset.get(idx) for idx in test_idxs]
print(f"Test idxs length: {len(test_idxs)}")
# print(test_idxs)

# Evaluate model on batch of training data and test data
# Train data
c = Collater.for_dataset(dataset, exclude_keys=[])
batch = c.collate(test_data_list)
print("Begin model evaluation on testing data...")
out = model(AtomicData.to_AtomicDataDict(batch))
atomic_energies = out[AtomicDataDict.PER_ATOM_ENERGY_KEY].detach().numpy()
total_energies = out[AtomicDataDict.TOTAL_ENERGY_KEY].detach().numpy()

aspirin_atoms = [
        "C1",
        "C2",
        "C3",
        "C4",
        "C5",
        "C6",
        "C7",
        "O1",
        "O2",
        "O3",
        "C8",
        "C9",
        "O4",
        "H1",
        "H2",
        "H3",
        "H4",
        "H5",
        "H6",
        "H7",
        "H8",
    ]

print(atomic_energies.shape)
for i in [3]:
    print(atomic_energies[i:len(atomic_energies):len(aspirin_atoms)].shape)
    T = 0.5
    x = np.linspace(0.0, N * T, N)
    y = atomic_energies[i:len(atomic_energies):len(aspirin_atoms)]
    xf = fftfreq(N, T)[:N//2]
    yf = fft(y)
    plt.grid()
    plt.plot(
        xf,
        2.0/N * np.abs(yf[:N//2]),
        label=aspirin_atoms[i]
    )

# plt.plot(
#     total_energies,
#     label="Total energy"
# )

plt.xlabel("Frequency (1/fs)", fontsize=18)
plt.ylabel("Relative Amplitude")
# ax.set_yscale("symlog")
plt.title("FFT of Carbon Atomic Energies of Aspirin")
plt.legend()
plt.savefig("aspirin_C_energies_fft.png")
