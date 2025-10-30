import os
import subprocess

network_paths = [
    # '/home/troyxdp/Documents/University Work/HYP/HYP Source Code/Back End/pytorch_networks/experiment_1/best_val_loss_network.pt',
    # '/home/troyxdp/Documents/University Work/HYP/HYP Source Code/Back End/pytorch_networks/experiment_2/best_val_loss_network.pt',
    '/home/troyxdp/Documents/University Work/HYP/HYP Source Code/Back End/pytorch_networks/experiment_3/best_val_loss_network.pt',
]

ncg_test_data_root = '/home/troyxdp/Documents/University Work/HYP/HYP Source Code/Back End/test_dataset/song_lists/ncg_tests'

experiment_paths = [
    # '/home/troyxdp/Documents/University Work/HYP/Experiment/PyTorch Statistics/NCG/Experiment 1',
    # '/home/troyxdp/Documents/University Work/HYP/Experiment/PyTorch Statistics/NCG/Experiment 2',
    '/home/troyxdp/Documents/University Work/HYP/Experiment/PyTorch Statistics/NCG/Experiment 3',
]

for i, network_path in enumerate(network_paths):
    experiment_path = experiment_paths[i]
    for ncg_test in sorted(os.listdir(ncg_test_data_root)):
        ncg_test_path = os.path.join(ncg_test_data_root, ncg_test)
        test_folder = f'Test {ncg_test.upper()} - NCG'
        output_path = os.path.join(experiment_path, test_folder)

        print(network_path)
        print(ncg_test_path)
        print(output_path)

        subprocess.run([
            "python3",
            "test_pytorch_model.py",
            "--pregenerated",
            "--output-path", output_path,
            "--pregenerated-lists-dataset-path", ncg_test_path,
            "--nn-path", network_path,
            "--traversal-algorithm", "optimal_path"
        ])

        print()