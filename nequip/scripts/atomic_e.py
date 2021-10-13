import torch
import numpy as np
from sklearn import mixture
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from ase.visualize import view
from sklearn.metrics import mean_absolute_error, mean_squared_error
from scipy import stats

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
test_idxs = [idx for idx in range(11000) if idx not in torch.cat((train_idxs, val_idxs)).tolist()]
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
for i in [0, 1, 2, 3, 4, 5, 6, 10, 11]:
    print(atomic_energies[i:len(atomic_energies):len(aspirin_atoms)].shape)
    plt.plot(
        atomic_energies[i:len(atomic_energies):len(aspirin_atoms)],
        label=aspirin_atoms[i]
    )

# plt.plot(
#     total_energies,
#     label="Total energy"
# )

plt.xlabel("Frame Number (After First 1000)", fontsize=18)
plt.ylabel("Energy (kcal/mol)")
ax.set_yscale("symlog")
plt.title("Carbon Atomic Energies of Aspirin")
plt.legend()
plt.savefig("aspirin_C_energies_10000.png")
