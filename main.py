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
    # number of neurons: 209 -> 128 -> 64 -> 128 -> 209
    nn = NeuralNetwork(
        input_size=209,
        output_size=209
    )
    # Layer 1
    layer_1_weights = np.random.rand(128, 209)
    layer_1_biases = np.random.rand(128)
    layer_1 = FeedForwardLayer(layer_1_weights, layer_1_biases, relu)
    # Layer 2
    layer_2_weights = np.random.rand(64, 128)
    layer_2_biases = np.random.rand(64)
    layer_2 = FeedForwardLayer(layer_2_weights, layer_2_biases, relu)
    # Layer 3
    layer_3_weights = np.random.rand(128, 64)
    layer_3_biases = np.random.rand(128)
    layer_3 = FeedForwardLayer(layer_3_weights, layer_3_biases, relu)
    # Layer 4
    layer_4_weights = np.random.rand(209, 128)
    layer_4_biases = np.random.rand(209)
    layer_4 = FeedForwardLayer(layer_4_weights, layer_4_biases, relu)
    # Add each layer
    nn.append_layer(layer_1)
    nn.append_layer(layer_2)
    nn.append_layer(layer_3)
    nn.append_layer(layer_4)

    # Create trainer object
    trainer = Trainer(
        training_network=nn,
        initial_lr=0.001,
        final_lr=0.0001,
        num_epochs=120,
        training_dataset_path='/home/troyxdp/Documents/University Work/HYP/HYP Source Code/MillionSongSubset'
    )

    # Demonstrate feed forward of network
    for item in trainer.get_data('/home/troyxdp/Documents/University Work/HYP/HYP Source Code/MillionSongSubset', '/home/troyxdp/Documents/University Work/HYP/HYP Source Code/MergedDataset/merged_dataset.csv'):
        # Check for missing values
        print()
        if trainer.is_missing_values(item):
            print(f"Song {item.song_name} is missing values")
            continue
        
        # Normalize song
        try:
            trainer.normalize_song(item)
        except ValueError:
            print(f"Error: invalid values in song {item.song_name} --- cannot normalize")
            continue

        # Run through neural network
        nn.set_input(item.get_nn_input())
        nn.feed_forward()
        output = nn.get_output()
        print(f"Successfully ran song {item.song_name} through pipeline")
        print(f"First 10 values in output: {output[:10]}")
