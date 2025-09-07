import os

import torch

from classes.pytorch_trainer import PyTorchTrainer
from classes.autoencoder import Autoencoder

if __name__ == '__main__':
    if input("Would you like to train an autoencoder using PyTorch? (y/n) ").lower() == 'y':
        # Create network
        autoencoder = Autoencoder()

        # Create trainer
        output_folder = input("Please input the folder you would like to save your models to: ")
        trainer = PyTorchTrainer(
            training_network=autoencoder,
            initial_lr=0.001,
            final_lr=0.00025,
            num_epochs=150,
            momentum=0.9,
            l2_regularization_lambda=0.001,
            dataset_path='./ProjectDataset',
            output_folder=output_folder,
        )

        # Check sureness that output path is what is actually desired
        if input(f"Are you sure you would like your model to be saved to {output_folder}? (y/n) ").lower() == 'y':
            opt_type = input("Would you like to use SGD or Adam optimization? (sgd/adam) ")
            trainer.train_model(opt_type=opt_type)
        else:
            print("Aborting training!")

    if input("\nWould you like to test the encoder section of a PyTorch model? (y/n) ").lower() == 'y':
        # Get path to weights
        nn_path = input("Please input the path to the neural network weights: ")
        if not os.path.exists(nn_path):
            print("Error: could not find network at path provided")
            exit(0)
        
        # Load network
        model = Autoencoder()
        model.load_state_dict(torch.load(nn_path, weights_only=True))

        # Get some file data
        trainer = PyTorchTrainer(
            training_network=model,
            initial_lr=0.001,
            final_lr=0.00025,
            num_epochs=10,
            momentum=0.9,
            l2_regularization_lambda=0,
            dataset_path='./ProjectDataset',
            output_folder='',
        )
        files = trainer.get_file_paths(input_dim=202, num_files=10)
        for file in files[:10]:
            song = trainer.get_song_data_from_file(file)
            x = torch.from_numpy(song.get_nn_input()).to(dtype=torch.float32)
            output = model.encode(x).detach().numpy()
            print(output)