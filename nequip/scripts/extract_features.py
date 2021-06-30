import torch

from nequip.utils import Config, dataset_from_config
from nequip.data import AtomicDataDict, AtomicData
from nequip.nn import SequentialGraphNetwork, SaveForOutput

path = "results/aspirin/minimal"

model = torch.load(path + "/best_model.pth")


# Find the sequential graph net (the bulk of the model):
def find_first_of_type(m: torch.nn.Module, kls) -> torch.nn.Module:
    if isinstance(m, kls):
        return m
    else:
        for child in m.children():
            tmp = find_first_of_type(child, kls)
            if tmp is not None:
                return tmp
    return None


sgn = find_first_of_type(model, SequentialGraphNetwork)

# Now insert a SaveForOutput
insert_after = "layer5_convnet"  # change this
sgn.insert_from_parameters(
    after=insert_after,
    name="feature_extractor",
    shared_params=dict(
        field=AtomicDataDict.NODE_FEATURES_KEY,
        out_field="saved"
    ),
    builder=SaveForOutput
)

# Load a config file
config = Config.from_file(path + "/config_final.yaml")
dataset = dataset_from_config(config)

# ...
data = dataset.get(0)
out = sgn(AtomicData.to_AtomicDataDict(data))

assert "saved" in out
print(out['saved'].shape)
