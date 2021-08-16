import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import argparse
import textwrap

from pathlib import Path
from nequip.utils import Config, dataset_from_config
from nequip.data import AtomicDataDict, AtomicData, Collater


def main(args=None):
    parser = argparse.ArgumentParser(
        description=textwrap.dedent(
            """Plot the final output block features of training data for a model.
            
            Usage: python .../plot_features.py train_dir chemical_system
            
            (Example: python plot_features.py results/aspirin/example-run aspirin)
            """
        )
    )
    parser.add_argument(
        "train_dir",
        help="Path to a working directory from a training session",
        type=Path
    )
    parser.add_argument(
        "chemical_system",
        help="The name of the chemical system on which the model is trained on (e.g. aspirin, 3BPA-300K, etc.)"
    )

    args = parser.parse_args(args=args)
    path = str(args.train_dir)
    system = str(args.chemical_system)

    # Load model and set in eval mode
    model = torch.load(path + "/best_model.pth", map_location=torch.device('cpu'))
    model.eval()

    # Load a config file
    config = Config.from_file(path + "/config_final.yaml")
    dataset = dataset_from_config(config)

    # Load trainer and get training and test data indexes and set up Collater
    trainer = torch.load(path + '/trainer.pth', map_location='cpu')
    train_idxs = trainer['train_idcs']

    # Create list of training data AtomicData objects
    data_list = [dataset.get(idx.item()) for idx in train_idxs]

    # Evaluate model on batch of training data and extract features from output
    c = Collater.for_dataset(dataset, exclude_keys=[])
    batch = c.collate(data_list)
    print("Begin model evaluation on training data...")
    out = model(AtomicData.to_AtomicDataDict(batch))
    features = out[AtomicDataDict.NODE_FEATURES_KEY].detach().numpy()

    # Get number of total atoms, feature length, and number of atoms in the chemical system
    tot_atoms, feature_length = features.shape
    num_atoms = tot_atoms // len(data_list)

    # Plot features per atom
    for atom in range(num_atoms):

        # Create dataframe of features
        atom_features = features[atom:tot_atoms:num_atoms]
        atom_features = atom_features.flatten()
        index = np.tile(np.arange(feature_length), len(data_list))
        df_atom_features = pd.DataFrame(atom_features, index=index, columns="Feature Value")

        # Plot figures
        plt.figure()
        plt.subplots(figsize=(16, 9))
        feature_plot = sns.histplot(
            df_atom_features,
            x=df_atom_features.index,
            y=df_atom_features.columns,
            bins=(np.arange(-0.5, 15.6, 1), np.arange(-0.45, 0.45, 0.05)),
            cbar=True,
            vmin=0,
            vmax=100,
            cmap="viridis",
        )
        feature_plot.set(xticks=np.arange(16), ylim=(-0.45, 0.45), yticks=np.arange(-0.45, 0.45, 0.05))
        plt.title(
            f"Atom {atom} Training Features ({system})",
            fontsize=18
        )
        plt.xlabel("Feature Index", fontsize=16)
        plt.savefig(
            f"atom{atom}_features_{system}.png"
        )


if __name__ == "__main__":
    main()
