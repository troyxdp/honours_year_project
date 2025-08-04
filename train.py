import os

import numpy as np

from classes.neural_network import NeuralNetwork, FeedForwardLayer
from classes.song import Song
from classes.trainer import Trainer

def relu(z_values: np.ndarray):
    to_ret = np.zeros(len(z_values))
    for i, z in enumerate(z_values):
        if z > 0:
            to_ret[i] = z
        else:
            to_ret[i] = 0
    return to_ret

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
        # # Layer 4
        # layer_4_weights = Trainer.get_he_initialization(64, 32)
        # layer_4_biases = np.random.rand(32) * bias_scale
        # layer_4 = FeedForwardLayer(layer_4_weights, layer_4_biases, NeuralNetwork.relu, NeuralNetwork.relu_dx)
        # # Layer 5
        # layer_5_weights = Trainer.get_he_initialization(32, 64)
        # layer_5_biases = np.random.rand(64) * bias_scale
        # layer_5 = FeedForwardLayer(layer_5_weights, layer_5_biases, NeuralNetwork.relu, NeuralNetwork.relu_dx)
        # Layer 6
        layer_6_weights = Trainer.get_he_initialization(64, 128)
        layer_6_biases = np.random.rand(128) * bias_scale
        layer_6 = FeedForwardLayer(layer_6_weights, layer_6_biases, NeuralNetwork.relu, NeuralNetwork.relu_dx)
        # Layer 7
        layer_7_weights = Trainer.get_he_initialization(128, 165)
        layer_7_biases = np.random.rand(165) * bias_scale
        layer_7 = FeedForwardLayer(layer_7_weights, layer_7_biases, NeuralNetwork.relu, NeuralNetwork.relu_dx)
        # Layer 8
        layer_8_weights = Trainer.get_he_initialization(165, 202)
        layer_8_biases = np.random.rand(202) * bias_scale
        layer_8 = FeedForwardLayer(layer_8_weights, layer_8_biases, NeuralNetwork.relu, NeuralNetwork.relu_dx)
        # Add each layer
        nn.append_layer(layer_1)
        nn.append_layer(layer_2)
        nn.append_layer(layer_3)
        # nn.append_layer(layer_4)
        # nn.append_layer(layer_5)
        nn.append_layer(layer_6)
        nn.append_layer(layer_7)
        nn.append_layer(layer_8)

        # Create trainer object
        trainer = Trainer(
            training_network=nn,
            initial_lr=0.001,
            final_lr=0.0001,
            num_epochs=150,
            dataset_path='./MillionSongSpotifyTracksDataset',
            output_folder='./networks/experiment_3'
        )

        trainer.train_model()

    if input("Would you like to extract the encoder section of a model? (y/n) ").lower() == 'y':
        nn = NeuralNetwork.load_network('./networks/experiment_2/best_val_loss_network.pkl')
        encoder = NeuralNetwork(202, 64)
        for i in range(int(nn.get_num_layers() / 2)):
            encoder.append_layer(nn.get_layer(i))
        print(encoder)
        encoder.save_network(os.path.join('./networks/encoder', 'encoder.pkl'))
