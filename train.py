import os

import numpy as np

from classes.neural_network import NeuralNetwork, FeedForwardLayer
from classes.song import Song
from classes.trainer import Trainer

if __name__ == '__main__':
    if input("Would you like to train a model? (y/n) ").lower() == 'y':
        # number of neurons: 202 -> 165 -> 128 -> 64 -> 32 -> 64 -> 128 -> 165 -> 202
        bias_scale = 0.0001
        nn = NeuralNetwork(
            input_size=202,
            output_size=202
        )
        # Layer 1
        layer_1_weights = Trainer.get_he_initialization(202, 165)
        layer_1_biases = np.random.rand(165) * bias_scale
        layer_1 = FeedForwardLayer(layer_1_weights, layer_1_biases, NeuralNetwork.relu, NeuralNetwork.relu_dx)
        # Layer 2
        layer_2_weights = Trainer.get_he_initialization(165, 128)
        layer_2_biases = np.random.rand(128) * bias_scale
        layer_2 = FeedForwardLayer(layer_2_weights, layer_2_biases, NeuralNetwork.relu, NeuralNetwork.relu_dx)
        # Layer 3
        layer_3_weights = Trainer.get_he_initialization(128, 64)
        layer_3_biases = np.random.rand(64) * bias_scale
        layer_3 = FeedForwardLayer(layer_3_weights, layer_3_biases, NeuralNetwork.relu, NeuralNetwork.relu_dx)
        # Layer 4
        layer_4_weights = Trainer.get_he_initialization(64, 128)
        layer_4_biases = np.random.rand(128) * bias_scale
        layer_4 = FeedForwardLayer(layer_4_weights, layer_4_biases, NeuralNetwork.relu, NeuralNetwork.relu_dx)
        # Layer 7
        layer_5_weights = Trainer.get_he_initialization(128, 165)
        layer_5_biases = np.random.rand(165) * bias_scale
        layer_5 = FeedForwardLayer(layer_5_weights, layer_5_biases, NeuralNetwork.relu, NeuralNetwork.relu_dx)
        # Layer 8
        layer_6_weights = Trainer.get_he_initialization(165, 202)
        layer_6_biases = np.random.rand(202) * bias_scale
        layer_6 = FeedForwardLayer(layer_6_weights, layer_6_biases, NeuralNetwork.relu, NeuralNetwork.relu_dx)
        # Add each layer
        nn.append_layer(layer_1)
        nn.append_layer(layer_2)
        nn.append_layer(layer_3)
        nn.append_layer(layer_4)
        nn.append_layer(layer_5)
        nn.append_layer(layer_6)

        # Create trainer object
        output_folder = input("Please input the folder you would like to save your models to: ")
        trainer = Trainer(
            training_network=nn,
            initial_lr=0.001,
            final_lr=0.00025,
            num_epochs=150,
            momentum=0.9,
            l2_regularization_lambda=0.001,
            dataset_path='./ProjectDataset',
            output_folder=output_folder,
        )

        if input(f"Are you sure you would like your model to be saved to {output_folder}? (y/n) ").lower() == 'y':
            trainer.train_model()
        else:
            print("Aborting training!")

    if input("\nWould you like to extract the encoder section of a model? (y/n) ").lower() == 'y':
        # Get path to neural network to turn into an encoder
        nn_path = input("Please input the path to the neural network weights: ")
        if not os.path.exists(nn_path):
            print("Error: could not find network at path provided")
            exit(0)

        # Load network and get encoder section
        nn : NeuralNetwork = NeuralNetwork.load_network(nn_path)
        encoder = NeuralNetwork(202, 64)
        for i in range(int(nn.get_num_layers() / 2)):
            encoder.append_layer(nn.get_layer(i))
        print(encoder)

        # Save encoder section
        nn_save_path = input("Please input the directory you would like to save the encoder to: ")
        if not os.path.isdir(os.path.split(nn_save_path)[0]):
            print("Error: could not find directory specified for saving")
            exit(0)
        encoder.save_network(os.path.join(nn_save_path, 'encoder.pkl'))