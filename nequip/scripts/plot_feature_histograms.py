"""Plot saved features for a trained aspirin model
Arguments: file atom features epoch

file: path to npz file containing saved hidden features
atom: atom in aspirin molecule to plot the hidden features for (integer 0-20)
features: which feature types to plot (0o, 0e, 1, 2, else plots all features)
epoch: epoch on which hidden features were saved"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import argparse


def main(args=None):
    arg_dict = vars(parse_command_line(args))
    file = arg_dict["file"]
    atom = int(arg_dict["atom"])
    features = arg_dict["features"]
    epoch = arg_dict["epoch"]
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
    num_atoms = len(aspirin_atoms)
    all_features = np.load(file)

    if features == "0o":
        bins = (np.arange(-0.5, 31.6, 1), np.arange(-1.5, 1.5, 0.15))
        feature_idx = range(32)
        vmax = 24
        ylim = (-1.5, 1.5)
        xticks = np.arange(32)
        yticks = np.arange(-1.5, 1.51, 0.3)
    elif features == "0e":
        bins = (np.arange(31.5, 63.6, 1), np.arange(-0.6, 0.6, 0.03))
        feature_idx = range(32, 64)
        vmax = 40
        ylim = (-0.6, 0.6)
        xticks = np.arange(32, 64)
        yticks = np.arange(-0.6, 0.6, 0.1)
    elif features == "1":
        bins = (np.arange(63.5, 159.6, 1), np.arange(-0.18, 0.18, 0.015))
        feature_idx = range(64, 160)
        vmax = 50
        ylim = (-0.18, 0.18)
        xticks = np.arange(64, 160, 6)
        yticks = np.arange(-0.18, 0.18, 0.03)
    elif features == "2":
        bins = (np.arange(159.5, 239.6, 1), np.arange(-0.15, 0.15, 0.015))
        feature_idx = range(160, 240)
        vmax = 50
        ylim = (-0.15, 0.15)
        xticks = np.arange(160, 240, 5)
        yticks = np.arange(-0.15, 0.15, 0.03)
    else:
        bins = (np.arange(-0.5, 239.6, 1), np.arange(-1.5, 1.5, 0.1))
        feature_idx = range(240)
        vmax = 50
        ylim = (-1.5, 1.5)
        xticks = np.arange(240, 10)
        yticks = np.arange(-1.5, 1.51, 0.3)

    data = np.empty((1, len(feature_idx)))
    for key in all_features.files:
        feature_batch = all_features[key]
        atoms_per_batch, _ = feature_batch.shape
        data = np.concatenate(
            (
                data,
                feature_batch[atom:atoms_per_batch:num_atoms][
                    :, feature_idx[0]: feature_idx[-1] + 1
                ],
            )
        )

    data = data[1:]
    index = np.tile(np.array(list(feature_idx)), data.shape[0])
    stack_data = data.flatten()
    df_features = pd.DataFrame(stack_data, index=index, columns=["Feature Value"])

    plt.subplots(figsize=(22, 9.6))
    feature_plot = sns.histplot(
        df_features,
        x=df_features.index,
        y=df_features.columns[0],
        bins=bins,
        cbar=True,
        vmin=0,
        vmax=vmax,
        cmap="viridis",
    )
    feature_plot.set(xticks=xticks, ylim=ylim, yticks=yticks)

    plt.title(
        f"{aspirin_atoms[atom]} Training Validation Feature Indices "
        f"{feature_idx[0]}-{feature_idx[-1]} Epoch {epoch} (Aspirin)"
    )
    plt.xlabel("Feature Index")
    plt.savefig(
        f"{aspirin_atoms[atom]}_epoch{epoch}_features{feature_idx[0]}-{feature_idx[-1]}"
    )


def parse_command_line(args=None):
    parser = argparse.ArgumentParser(
        description="Plot 2D histogram of saved hidden features of a model during training"
    )
    parser.add_argument("file", help="hidden features npz file path")
    parser.add_argument(
        "atom",
        help="which atom to plot hidden features for, must be an integer 0-20 for aspirin",
    )
    parser.add_argument("features", help="choose 0o, 0e, 1, 2, or all")
    parser.add_argument("epoch", help="epoch on which features were saved")
    args = parser.parse_args(args=args)
    return args


if __name__ == "__main__":
    main()
