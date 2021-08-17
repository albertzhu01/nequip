import torch
import numpy as np
from sklearn import mixture
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error

from nequip.utils import Config, dataset_from_config
from nequip.data import AtomicDataDict, AtomicData, Collater


def main():
    path_300K = "/n/home10/axzhu/nequip/results/bpa/train300K_072321"
    path_mixed = "/n/home10/axzhu/nequip/results/bpa_mixed/train_mixed_072721"

    model_300K = torch.load(path_300K + "/best_model.pth", map_location=torch.device('cpu'))
    model_mixed = torch.load(path_mixed + "/best_model.pth", map_location=torch.device('cpu'))
    model_300K.eval()
    model_mixed.eval()

    # Load config files
    config_300K_train = Config.from_file(path_300K + "/config_final.yaml")
    dataset_300K_train = dataset_from_config(config_300K_train)

    config_mixed_train = Config.from_file(path_mixed + "/config_final.yaml")
    dataset_mixed_train = dataset_from_config(config_mixed_train)

    config_300K_test = Config.from_file("/n/home10/axzhu/nequip/configs/bpa_300K.yaml")
    dataset_300K_test = dataset_from_config(config_300K_test)

    config_600K_test = Config.from_file("/n/home10/axzhu/nequip/configs/bpa_600K.yaml")
    dataset_600K_test = dataset_from_config(config_600K_test)

    config_1200K_test = Config.from_file("/n/home10/axzhu/nequip/configs/bpa_1200K.yaml")
    dataset_1200K_test = dataset_from_config(config_1200K_test)

    # Load trainers and get training and test data indexes and set up Collater
    trainer_300K = torch.load(path_300K + '/trainer.pth', map_location='cpu')
    train_idxs_300K = trainer_300K['train_idcs']
    val_idxs_300K = trainer_300K['val_idcs']

    trainer_mixed = torch.load(path_mixed + '/trainer.pth', map_location='cpu')
    train_idxs_mixed = trainer_mixed['train_idcs']
    val_idxs_mixed = trainer_mixed['val_idcs']
    # print(f"# of training points: {len(train_idxs)}")
    # print(f"# of val points: {len(val_idxs)}")
    # val_idxs = trainer['val_idcs']
    # print(val_idxs)
    # test_idxs = [idx for idx in range(len(dataset)) if idx not in train_idxs]

    def evaluate(dataset, data_list, test_train, model):
        c = Collater.for_dataset(dataset, exclude_keys=[])
        if test_train == 'test':
            batch = c.collate(data_list[:100])
        else:
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

    # Create list of training and test data AtomicData objects
    def get_logprobs_maes(train_dataset, train_idxs, val_idxs, test_dataset, model):
        train_data_list = [train_dataset.get(idx.item()) for idx in train_idxs]
        train_val_data_list = [train_dataset.get(idx.item()) for idx in torch.cat((train_idxs, val_idxs))]
        train_data_atoms = [atomic_data.to_ase() for atomic_data in train_val_data_list]
        test_data_list = [test_dataset.get(idx) for idx in range(len(test_dataset))
                          if test_dataset.get(idx).to_ase() not in train_data_atoms]

        print(f"Train + val dataset length: {len(train_val_data_list)}")
        print(f"Test dataset length: {len(test_data_list)}")

        train_features, train_force_maes = evaluate(train_dataset, train_data_list, "train", model)
        test_features, test_force_maes = evaluate(test_dataset, test_data_list, "test", model)

        test_tot_atoms, _ = test_features.shape
        print(f"total test atoms: {test_tot_atoms}")

        return train_features, test_features, test_force_maes, test_tot_atoms

    tr_300K_feat, tr_300K_te_300K_feat, tr_300K_te_300K_mae, tr_300K_te_300K_atoms = get_logprobs_maes(
        dataset_300K_train,
        train_idxs_300K,
        val_idxs_300K,
        dataset_300K_test,
        model_300K
    )

    _, tr_300K_te_600K_feat, tr_300K_te_600K_mae, tr_300K_te_600K_atoms = get_logprobs_maes(
        dataset_300K_train,
        train_idxs_300K,
        val_idxs_300K,
        dataset_600K_test,
        model_300K
    )

    _, tr_300K_te_1200K_feat, tr_300K_te_1200K_mae, tr_300K_te_1200K_atoms = get_logprobs_maes(
        dataset_300K_train,
        train_idxs_300K,
        val_idxs_300K,
        dataset_1200K_test,
        model_300K
    )

    tr_mix_feat, tr_mix_te_300K_feat, tr_mix_te_300K_mae, tr_mix_te_300K_atoms = get_logprobs_maes(
        dataset_mixed_train,
        train_idxs_mixed,
        val_idxs_mixed,
        dataset_300K_test,
        model_mixed
    )

    _, tr_mix_te_600K_feat, tr_mix_te_600K_mae, tr_mix_te_600K_atoms = get_logprobs_maes(
        dataset_mixed_train,
        train_idxs_mixed,
        val_idxs_mixed,
        dataset_600K_test,
        model_mixed
    )

    _, tr_mix_te_1200K_feat, tr_mix_te_1200K_mae, tr_mix_te_1200K_atoms = get_logprobs_maes(
        dataset_mixed_train,
        train_idxs_mixed,
        val_idxs_mixed,
        dataset_1200K_test,
        model_mixed
    )

    gmm_300K = mixture.GaussianMixture(n_components=24, covariance_type='full', random_state=0)
    gmm_300K.fit(tr_300K_feat)
    print(f"GMM_300K converged? {gmm_300K.converged_}")

    gmm_mixed = mixture.GaussianMixture(n_components=24, covariance_type='full', random_state=0)
    gmm_mixed.fit(tr_mix_feat)
    print(f"GMM_mixed converged? {gmm_mixed.converged_}")

    num_atoms = 27

    # Plot force MAE and RMSE vs. relative confidence for each atom
    percentiles = np.arange(0, 100, 5)

    def compute_maes(logprobs, maes):
        f_maes = []
        for p in np.nditer(percentiles):
            cutoff_idxs = np.argwhere(logprobs >= np.percentile(logprobs, p)).reshape(-1)
            f_maes.append(np.mean(maes[cutoff_idxs]))
        return f_maes

    for i in range(num_atoms):
        atom_logprobs_300K_300K = gmm_300K.score_samples(tr_300K_te_300K_feat[i:tr_300K_te_300K_atoms:num_atoms])
        atom_force_maes_300K_300K = tr_300K_te_300K_mae[i:tr_300K_te_300K_atoms:num_atoms]
        f_maes_300K_300K = compute_maes(atom_logprobs_300K_300K, atom_force_maes_300K_300K)

        atom_logprobs_300K_600K = gmm_300K.score_samples(tr_300K_te_600K_feat[i:tr_300K_te_600K_atoms:num_atoms])
        atom_force_maes_300K_600K = tr_300K_te_600K_mae[i:tr_300K_te_600K_atoms:num_atoms]
        f_maes_300K_600K = compute_maes(atom_logprobs_300K_600K, atom_force_maes_300K_600K)

        atom_logprobs_300K_1200K = gmm_300K.score_samples(tr_300K_te_1200K_feat[i:tr_300K_te_1200K_atoms:num_atoms])
        atom_force_maes_300K_1200K = tr_300K_te_1200K_mae[i:tr_300K_te_1200K_atoms:num_atoms]
        f_maes_300K_1200K = compute_maes(atom_logprobs_300K_1200K, atom_force_maes_300K_1200K)

        atom_logprobs_mix_300K = gmm_mixed.score_samples(tr_mix_te_300K_feat[i:tr_mix_te_300K_atoms:num_atoms])
        atom_force_maes_mix_300K = tr_mix_te_300K_mae[i:tr_mix_te_300K_atoms:num_atoms]
        f_maes_mix_300K = compute_maes(atom_logprobs_mix_300K, atom_force_maes_mix_300K)

        atom_logprobs_mix_600K = gmm_mixed.score_samples(tr_mix_te_600K_feat[i:tr_mix_te_600K_atoms:num_atoms])
        atom_force_maes_mix_600K = tr_mix_te_600K_mae[i:tr_mix_te_600K_atoms:num_atoms]
        f_maes_mix_600K = compute_maes(atom_logprobs_mix_600K, atom_force_maes_mix_600K)

        atom_logprobs_mix_1200K = gmm_mixed.score_samples(tr_mix_te_1200K_feat[i:tr_mix_te_1200K_atoms:num_atoms])
        atom_force_maes_mix_1200K = tr_mix_te_1200K_mae[i:tr_mix_te_1200K_atoms:num_atoms]
        f_maes_mix_1200K = compute_maes(atom_logprobs_mix_1200K, atom_force_maes_mix_1200K)

        plt.figure()
        plt.subplots(figsize=(16, 9))
        plt.plot(
            percentiles,
            f_maes_300K_300K,
            color='blue',
            marker='o',
            label=f'300K Train, 300K Test'
        )
        plt.plot(
            percentiles,
            f_maes_300K_600K,
            color='green',
            marker='o',
            label=f'300K Train, 600K Test'
        )
        plt.plot(
            percentiles,
            f_maes_300K_1200K,
            color='red',
            marker='o',
            label=f'300K Train, 1200K Test'
        )
        plt.plot(
            percentiles,
            f_maes_mix_300K,
            color='deepskyblue',
            marker='o',
            label=f'Mixed-T Train, 300K Test'
        )
        plt.plot(
            percentiles,
            f_maes_mix_600K,
            color='lawngreen',
            marker='o',
            label=f'Mixed-T Train, 600K Test'
        )
        plt.plot(
            percentiles,
            f_maes_mix_1200K,
            color='lightcoral',
            marker='o',
            label=f'Mixed-T Train, 1200K Test'
        )
        plt.legend(fontsize=14)
        plt.title(f"3BPA Atom Index {i} Error vs. Relative Confidence", fontsize=18)
        plt.xlabel("Confidence Percentile", fontsize=16)
        plt.ylabel("Force MAE (eV/A)", fontsize=16)
        plt.savefig(f"bpa_atom{i}_err_vs_rel-conf.png")


if __name__ == "__main__":
    main()
