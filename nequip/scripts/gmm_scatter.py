import torch
import numpy as np
import matplotlib.pyplot as plt
import argparse
import textwrap

from pathlib import Path
from sklearn import mixture
from sklearn.metrics import mean_absolute_error
from scipy import stats
from nequip.utils import Config, dataset_from_config
from nequip.data import AtomicDataDict, AtomicData, Collater


def main(args=None):
    parser = argparse.ArgumentParser(
        description=textwrap.dedent(
            """Plot log-probability density (obtained by GMM) vs. force MAE for each atom in a chemical system
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
    parser.add_argument(
        "dataset_config_test",
        help="Path to a YAML config file (including the file) specifying the dataset to load test data from",
        type=Path,
        default=None
    )
    parser.add_argument(
        "--train_temp",
        help="Temperature of training data",
        type=str,
        default="300K"
    )
    parser.add_argument(
        "--test_temp",
        help="Temperature of test data",
        type=str,
        default="300K"
    )

    args = parser.parse_args(args=args)
    path = str(args.train_dir)
    system = str(args.chemical_system)
    train_temp = args.train_temp
    test_temp = args.test_temp

    # Load model
    model = torch.load(path + "/best_model.pth", map_location=torch.device('cpu'))
    model.eval()

    # Load train and test config files
    config_train = Config.from_file(path + "/config_final.yaml")
    dataset_train = dataset_from_config(config_train)
    config_test = Config.from_file(str(args.dataset_config_test))
    dataset_test = dataset_from_config(config_test)

    # Load trainer and get training and val data indexes
    trainer = torch.load(path + '/trainer.pth', map_location='cpu')
    train_idxs = trainer['train_idcs']
    val_idxs = trainer['val_idcs']
    print(f"# of training points: {len(train_idxs)}")
    print(f"# of val points: {len(val_idxs)}")

    # Create list of training and test data, ensuring that the the test data does not contain train or val data
    train_data_list = [dataset_train.get(idx.item()) for idx in train_idxs]
    train_val_data_list = [dataset_train.get(idx.item()) for idx in torch.cat((train_idxs, val_idxs))]
    train_data_atoms = [atomic_data.to_ase() for atomic_data in train_val_data_list]
    test_data_list = [dataset_test.get(idx) for idx in range(len(dataset_test))
                      if dataset_test.get(idx).to_ase() not in train_data_atoms]
    test_data_atoms = [atomic_data.to_ase() for atomic_data in test_data_list]
    test_idxs = [idx for idx in range(len(dataset_test)) if dataset_test.get(idx).to_ase() in test_data_atoms]
    print(f"Train + val dataset length: {len(train_val_data_list)}")
    print(f"Test dataset length: {len(test_data_list)}")
    print(f"Test idxs length: {len(test_idxs)}")

    # Function to evaluate model on data
    def evaluate(dataset, data_list, test_train):
        c = Collater.for_dataset(dataset, exclude_keys=[])
        batch = c.collate(data_list)
        print(f"Begin model evaluation on {test_train} data...")
        out = model(AtomicData.to_AtomicDataDict(batch))
        features = out[AtomicDataDict.NODE_FEATURES_KEY].detach().numpy()
        pred_forces = out[AtomicDataDict.FORCE_KEY].detach().numpy()
        a_forces = np.array([atomic_data.forces.detach().numpy() for atomic_data in data_list])
        actual_forces = a_forces.reshape(-1, a_forces.shape[-1])
        print(f"{test_train}_pred_forces shape: {pred_forces.shape}")
        print(f"{test_train}_actual_forces shape: {actual_forces.shape}")
        force_maes = []
        for force in range(len(pred_forces)):
            force_maes.append(mean_absolute_error(pred_forces[force], actual_forces[force]))
        force_maes = np.array(force_maes)
        return features, force_maes

    # Evaluate model on batch of training data and test data
    train_features, train_force_maes = evaluate(dataset_train, train_data_list, "train")
    test_features, test_force_maes = evaluate(dataset_test, test_data_list, "test")

    # Get dimensions of train and test features and number of atoms in aspirin
    train_tot_atoms, feature_length = train_features.shape
    num_atoms = train_tot_atoms // len(train_data_list)
    test_tot_atoms, _ = test_features.shape
    print(f"num_atoms: {num_atoms}")
    print(f"total train atoms: {train_tot_atoms}")
    print(f"total test atoms: {test_tot_atoms}")

    # Train GMM on training features
    n_components = np.arange(1, 28)
    models = [mixture.GaussianMixture(n_components=n, covariance_type='full', random_state=0) for n in n_components]
    bics = [model.fit(train_features).bic(train_features) for model in models]
    print(f"Number of components with min BIC: {bics.index(min(bics))}")
    gmm = mixture.GaussianMixture(n_components=bics.index(min(bics)), covariance_type='full', random_state=0)
    gmm.fit(train_features)
    print(f"GMM converged? {gmm.converged_}")

    # Make scatterplot of log-prob vs. force MAE for train and test data for one atom
    for i in range(num_atoms):

        # Training data force MAEs and log-prob densities for 1 atom
        atom_i_train_force_maes = train_force_maes[i:train_tot_atoms:num_atoms]
        atom_i_train_log_probs = gmm.score_samples(train_features[i:train_tot_atoms:num_atoms])

        # 'Not-so-arbitrary' cut-offs for chemical accuracy and uncertainty
        mae_cutoff = 0.043
        logprob_cutoff = np.percentile(atom_i_train_log_probs, 2.5)

        # Bad (above mae_cutoff) testing data force MAEs and log-prob densities for 1 atom
        atom_i_test_force_maes = test_force_maes[i:test_tot_atoms:num_atoms]
        atom_i_bad_test_maes_idx = np.where(atom_i_test_force_maes > mae_cutoff)
        atom_i_bad_test_maes = atom_i_test_force_maes[atom_i_bad_test_maes_idx]
        atom_i_test_log_probs = gmm.score_samples(test_features[i:test_tot_atoms:num_atoms])
        atom_i_bad_test_logprobs = atom_i_test_log_probs[atom_i_bad_test_maes_idx]

        # Good (below mae_cutoff) testing data force MAEs and log-prob densities for 1 atom
        atom_i_good_test_mae_idx = np.setdiff1d(np.arange(len(test_data_list)), atom_i_bad_test_maes_idx)
        atom_i_good_test_maes = atom_i_test_force_maes[atom_i_good_test_mae_idx]
        atom_i_good_test_logprobs = atom_i_test_log_probs[atom_i_good_test_mae_idx]

        # r correlation for all test data
        test_r, _ = stats.pearsonr(atom_i_test_force_maes, atom_i_test_log_probs)

        # Number of good and bad test data points, number of each below log-prob cutoff
        num_test_bad_mae = len(atom_i_bad_test_maes)
        num_test_good_mae = len(atom_i_good_test_maes)
        num_test_bad_logprob = np.where(atom_i_bad_test_logprobs < logprob_cutoff)[0].size
        num_below_l_cutoff = np.where(atom_i_test_log_probs < logprob_cutoff)[0].size
        num_test_good_logprob = num_below_l_cutoff - num_test_bad_logprob

        # Plot everything
        plt.figure()
        plt.subplots(figsize=(16, 9))
        plt.rc('xtick', labelsize=14)
        plt.rc('ytick', labelsize=14)
        plt.scatter(
            x=atom_i_good_test_maes,
            y=atom_i_good_test_logprobs,
            color='b',
            label=f'Test {test_temp} good ({num_test_good_logprob}/{num_test_good_mae})'
        )
        plt.scatter(
            x=atom_i_bad_test_maes,
            y=atom_i_bad_test_logprobs,
            color='r',
            label=f'Test {test_temp} bad ({num_test_bad_logprob}/{num_test_bad_mae})'
        )
        plt.scatter(
            x=atom_i_train_force_maes,
            y=atom_i_train_log_probs,
            color='k',
            label=f'Train {train_temp}'
        )
        plt.axhline(
            logprob_cutoff,
            color='k',
            linestyle='--',
            label='Uncertainty cutoff (2.5th percentile of training data)'
        )
        plt.axvline(mae_cutoff, color='m', linestyle='--', label='Chemical accuracy cutoff')
        plt.plot([], [], ' ', label=f"All test data: r = {test_r:.3f}")
        plt.legend(fontsize=14)
        plt.title(
            f"Atom Index {i} Log-Probability Density vs. Force MAE (Train {train_temp}, Test {test_temp})",
            fontsize=18
        )
        plt.xlabel("Force MAE (eV/A)", fontsize=16)
        plt.ylabel("Log-Probability Density", fontsize=16)
        plt.savefig(f"atom{i}_logprob_vs_mae_{system}.png")


if __name__ == "__main__":
    main()
