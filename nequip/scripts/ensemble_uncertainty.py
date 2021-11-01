import torch
import numpy as np
import matplotlib.pyplot as plt
import argparse
import textwrap

from pathlib import Path
from sklearn.metrics import mean_absolute_error
from nequip.utils import Config, dataset_from_config
from nequip.data import AtomicDataDict, AtomicData, Collater


def main(args=None):
    parser = argparse.ArgumentParser(
        description=textwrap.dedent(
            """Save npz files for atomic energy predictions, atomic energy MAEs,
               atomic force predictions, and atomic force MAEs for a nequip model
            """
        )
    )
    parser.add_argument(
        "train_dir",
        help="Path to a working directory from a training session",
        type=Path
    )
    # parser.add_argument(
    #     "chemical_system",
    #     help="The name of the chemical system on which the model is trained on (e.g. aspirin, 3BPA-300K, etc.)"
    # )
    parser.add_argument(
        "dataset_config_test",
        help="Path to a YAML config file (including the file) specifying the dataset to load test data from",
        type=Path,
        default=None
    )

    args = parser.parse_args(args=args)
    path = str(args.train_dir)
    # system = str(args.chemical_system)
    # train_temp = args.train_temp
    # test_temp = args.test_temp

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
    train_data_list = [dataset_train.get(idx) for idx in train_idxs]
    train_val_data_list = [dataset_train.get(idx) for idx in train_idxs + val_idxs]
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
        pred_energies = out[AtomicDataDict.PER_ATOM_ENERGY_KEY].detach().numpy()
        pred_forces = out[AtomicDataDict.FORCE_KEY].detach().numpy()
        a_forces = np.array([atomic_data.forces.detach().numpy() for atomic_data in data_list])
        actual_forces = a_forces.reshape(-1, a_forces.shape[-1])
        print(f"{test_train}_pred_energies shape: {pred_energies.shape}")
        print(f"{test_train}_pred_forces shape: {pred_forces.shape}")
        print(f"{test_train}_actual_forces shape: {actual_forces.shape}")
        force_maes = []
        for force in range(len(pred_forces)):
            force_maes.append(mean_absolute_error(pred_forces[force], actual_forces[force]))
        force_maes = np.array(force_maes)
        return [pred_energies, pred_forces, force_maes]

    # Evaluate model on batch of training data and test data
    train_out_e_f = evaluate(dataset_train, train_data_list, "train")
    test_out_e_f = evaluate(dataset_test, test_data_list, "test")

    np.savez(f'train_atomic_e_{path[-9:]}_{str(args.dataset_config_test)[-9:-5]}', train_out_e_f[0])
    np.savez(f'train_forces_{path[-9:]}_{str(args.dataset_config_test)[-9:-5]}', train_out_e_f[1])
    np.savez(f'train_forces_mae_{path[-9:]}_{str(args.dataset_config_test)[-9:-5]}', train_out_e_f[2])

    np.savez(f'test_atomic_e_{path[-9:]}_{str(args.dataset_config_test)[-9:-5]}', test_out_e_f[0])
    np.savez(f'test_forces_{path[-9:]}_{str(args.dataset_config_test)[-9:-5]}', test_out_e_f[1])
    np.savez(f'test_forces_mae_{path[-9:]}_{str(args.dataset_config_test)[-9:-5]}', test_out_e_f[2])


if __name__ == "__main__":
    main()
