import numpy as np
import torch
import nequip
from ase.io import read
from nequip.data import AtomicData, AtomicDataDict
from nequip.scripts import deploy
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

atoms = read("C:/Users/alber/nequip/nequip/scripts/aspirin.xyz")
device = "cuda" if torch.cuda.is_available() else "cpu"
model_path = "C:/Users/alber/nequip/nequip/scripts/aspirin_50_epochs_new/results/aspirin/example-run/deployed.pth"
model, metadata = deploy.load_deployed_model(model_path=model_path, device="cpu")
r_max = float(metadata[nequip.scripts.deploy.R_MAX_KEY])
data = AtomicData.from_ase(atoms=atoms, r_max=r_max)
out = model(AtomicData.to_AtomicDataDict(data))

hidden_features = out['feature_vectors'].detach().cpu().numpy()

plt.plot(hidden_features[0])
plt.show()
