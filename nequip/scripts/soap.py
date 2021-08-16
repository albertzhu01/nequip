import torch
import matplotlib.pyplot as plt
from sklearn.metrics import pairwise
import argparse
import textwrap

from pathlib import Path
from nequip.utils import Config, dataset_from_config
from nequip.data import AtomicDataDict, AtomicData, Collater

# Set seed
import random
random.seed(0)

# Import SOAP
from dscribe.descriptors import SOAP


def main(args=None):
    parser = argparse.ArgumentParser(
        description=textwrap.dedent(
            """Plot feature Euclidean distance vs. SOAP output cosine distance for atoms in a chemical system.
            """
        )
    )
    parser.add_argument(
        "train_dir",
        help="Path to a working directory from a training session",
        type=Path
    )
    parser.add_argument(
        "species_list",
        help="List of chemical elements present in the chemical system. Must be a list of atomic symbols (e.g. C H)",
        nargs="*"
    )
    parser.add_argument(
        "chemical_system",
        help="The name of the chemical system on which the model is trained on (e.g. aspirin, 3BPA, etc.)"
    )
    parser.add_argument(
        "--dataset_config",
        help="Path to a YAML config file specifying the dataset to load test data from. If omitted, `config_final.yaml` in `train_dir` will be used",
        type=Path,
        default=None
    )
    parser.add_argument(
        "--rcut",
        help="Radius cutoff to be used in the SOAP descriptor.",
        type=float,
        default=6.0
    )
    parser.add_argument(
        "--nmax",
        help="nmax value to be used in the SOAP descriptor",
        type=int,
        default=8
    )
    parser.add_argument(
        "--lmax",
        help="lmax value to be used int the SOAP descriptor",
        type=int,
        default=6
    )
    parser.add_argument(
        "--periodic",
        help="periodic condition to be used int the SOAP descriptor",
        type=bool,
        default=False
    )

    args = parser.parse_args(args=args)
    path = str(args.train_dir)
    system = str(args.chemical_system)

    # Setting up the SOAP descriptor
    soap = SOAP(
        species=args.species_list,
        periodic=args.periodic,
        rcut=args.rcut,
        nmax=args.nmax,
        lmax=args.lmax,
    )

    if args.dataset_config is not None:
        config = str(args.dataset_config)
        dataset = dataset_from_config(config)
        test_idxs = list(range(len(dataset)))
    else:
        config = Config.from_file(path + "/config_final.yaml")
        dataset = dataset_from_config(config)
        trainer = torch.load(path + '/trainer.pth', map_location='cpu')
        train_idxs = trainer['train_idcs']
        val_idxs = trainer['val_idcs']
        test_idxs = [idx for idx in range(len(dataset)) if idx not in torch.cat((train_idxs, val_idxs))]

    random.shuffle(test_idxs)
    sample_idxs = test_idxs[:100]
    print(f"Sample idxs: {sample_idxs}")
    test_sample = [dataset.get(idx) for idx in sample_idxs]

    # Load model
    model = torch.load(path + "/best_model.pth", map_location=torch.device('cpu'))
    model.eval()

    # Create ASE.Atoms from test set
    bpa_list = []
    for atomic_data in test_sample:
        bpa_list.append(atomic_data.to_ase())

    # Create SOAP output for chemical system
    soap_sys = soap.create(bpa_list)
    print(f"soap shape: {soap_sys.shape}")
    print(f"soap shape for one atom: {soap_sys[:, 0, :].shape}")

    # Evaluate model on test data samples and extract atomic features
    c_test = Collater.for_dataset(dataset, exclude_keys=[])
    test_batch = c_test.collate(test_sample)
    print("Begin model evaluation on test data...")
    test_out = model(AtomicData.to_AtomicDataDict(test_batch))
    test_features = test_out[AtomicDataDict.NODE_FEATURES_KEY]
    print(f"Atomic features shape: {test_features.shape}")

    # Get dimensions of train and test features and number of atoms in molecule
    test_sample_len = len(test_sample)
    test_tot_atoms, feature_length = test_features.shape
    num_atoms = test_tot_atoms // test_sample_len
    print(f"Total test atoms per sample: {test_tot_atoms}")
    print(f"Number of atoms in molecule: {num_atoms}")

    # Next compute the atomic input and feature distances per atom for all 100 molecules in the 2 test samples
    for i in range(num_atoms):
        soap_atom_i = torch.tensor(soap_sys[:, i, :])
        print(f"Atom {i} SOAP shape: {soap_atom_i.shape}")

        soap_dists = pairwise.cosine_distances(soap_atom_i, soap_atom_i).reshape(-1)
        print(f"Atom {i} SOAP distances shape: {soap_dists.shape}")

        feature_dists = torch.cdist(
            test_features[i:test_tot_atoms:num_atoms].view(1, test_sample_len, feature_length),
            test_features[i:test_tot_atoms:num_atoms].view(1, test_sample_len, feature_length),
            p=2
        ).view(-1)

        print(f"Feature distances shape: {feature_dists.shape}")

        # Plot
        plt.figure()
        plt.subplots(figsize=(16, 9))
        plt.scatter(
            x=soap_dists,
            y=feature_dists.detach().numpy(),
        )
        plt.title(f"Atom Index {i} Feature Distance vs. SOAP Distance ({system})", fontsize=18)
        plt.xlabel("SOAP Distance", fontsize=16)
        plt.ylabel("Feature Distance", fontsize=16)
        plt.savefig(f"atom{i}_soap-dist_{system}.png")


if __name__ == "__main__":
    main()
