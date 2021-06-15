import numpy as np
import torch
from nequip.data import AtomicData, AtomicDataDict
from nequip.utils import Config
from nequip.models import ForceModel

aspirin_data = np.load("C:/Users/alber/nequip/nequip/scripts/benchmark_data/aspirin_ccsd-test.npz")

r = aspirin_data['R'][-1]

model_path = "C:/Users/alber/nequip/nequip/scripts/results/aspirin/example-run/best_model.pth"
config = Config.from_file("C:/Users/alber/nequip/configs/example.yaml")
final_model = ForceModel(**dict(config))
final_model.load_state_dict(torch.load(model_path))
final_model.eval()

data = AtomicData.from_points(
    pos=r,
    r_max=4.0,
    **{AtomicDataDict.ATOMIC_NUMBERS_KEY:
           torch.Tensor(torch.from_numpy(aspirin_data['z'].astype(np.float32))).to(torch.int64)}
)

pred = final_model(AtomicData.to_AtomicDataDict(data))['feature_vectors']

print(pred)
