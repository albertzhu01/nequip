import torch

from nequip.utils import Config, dataset_from_config
from nequip.data import AtomicDataDict, AtomicData, Collater
from nequip.nn import SequentialGraphNetwork, SaveForOutput

# path = "C:/Users/alber/nequip/nequip/scripts/aspirin_50_epochs_new/results/aspirin/example-run"
path = "/n/home10/axzhu/results/aspirin/example-run"

model = torch.load(path + "/best_model.pth", map_location=torch.device('cpu'))


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

# Load trainer and get training data indexes
trainer = torch.load(path + '/trainer.pth', map_location='cpu')
train_idxs = trainer['train_idcs']

# Create list of training data AtomicData objects
data_list = [dataset.get(idx.item()) for idx in train_idxs]
print(len(data_list))

# Evaluate model on batch of training data
c = Collater.for_dataset(dataset)
batch = c.collate(data_list)
out = model(AtomicData.to_AtomicDataDict(batch))
assert AtomicDataDict.NODE_FEATURES_KEY in out
print(out[AtomicDataDict.NODE_FEATURES_KEY].shape)
