import os
import pickle

import numpy as np


# TODO: add error checking for each of the methods
# TODO: implement pretty printing of neural network and each layer


# Activation functions

class Layer():
    
    def __init__(
        self
    ):
        self._input = None
        self._output = None

    def forward(self):
        raise NotImplementedError
    
    def update_layer(self, grad):
        raise NotImplementedError
    
    def set_input(self, input:np.ndarray):
        self._input = input

    def get_input(self):
        return self._input

    def get_output(self):
        return self._output
    
    def contains_nan(self):
        raise NotImplementedError

class FeedForwardLayer(Layer):

    def __init__(
        self,
        weights:np.ndarray=None,
        bias:np.ndarray=None,
        activation_function=None,
        activation_funcion_dx=None
    ):
        # Initialize superclass
        super().__init__()

        # Initialize forwarding properties
        self._weights = weights
        self._bias = bias
        self._activation_function = activation_function

        # Initialize backpropogation properties
        self._activation_function_dx = activation_funcion_dx
        self._velocity_weights = None
        self._velocity_bias = None

        # Initialize forwarding results properties
        self._z_values = None

        # Set properties for dimensions
        self.num_inputs = None
        self.num_outputs = None
        if not self._weights is None:
            # X: 3 x 4 . y: 4 x 1 = 3 x 1 because X.q = y.p. We are trying to go from 4 inputs to 3 outputs in this example
            # X: 4 x 3 . y: 3 x 1 = 4 x 1 because X.q = y.p. We are trying to go from 3 inputs to 4 outputs in this example
            self.num_inputs = self._weights.shape[1] # number of inputs is number of columns
            self.num_outputs = self._weights.shape[0] # number of outputs is number of rows 

        # Check bias was added as well if weights were added
        if not self._weights is None:
            if self._bias is None:
                raise ValueError("Error: weights are set but bias is not")

        # Create inputs and outputs
        if not self.num_inputs is None:
            self._input = np.zeros(self.num_inputs)
            self._output = np.zeros(self.num_outputs)

    # SETTER METHODS
    def set_weights(self, weights:np.ndarray):
        # Check dimensions of inputted weights if self._weights is already set
        if not self._weights is None:
            if not self._weights.shape == weights.shape:
                raise ValueError("Error: dimensions of current weights do not match dimensions of inputted weights")
            
        # Check if there are any NaNs in inputted weights
        if np.any(np.isnan(weights)):
            raise ValueError("Error: inputted weights contain at least one NaN value")
        
        # Set weights
        self._weights = weights

    def set_biases(self, bias:np.ndarray):
        # Check dimensions if self._biases is already set
        if not self._bias is None:
            if not self._bias.shape == bias.shape:
                raise ValueError("Error: length of current bias vector does not match length of inputted bias vector")
            
        # Check for NaN values
        if np.any(np.isnan(bias)):
            raise ValueError("Error: inputted bias vector contains at least one NaN value")
        
        # Set bias
        self._bias = bias

    def set_activation_function(self, act_fn):
        self._activation_function = act_fn

    def update_velocity(self, weights_grad, bias_grad, momentum, lr):
        # Set velocity value if it has not been initialized
        if self._velocity_weights is None and self._velocity_bias is None:
            self._velocity_weights = np.zeros_like(weights_grad, dtype=np.float64)
            self._velocity_bias = np.zeros_like(bias_grad, dtype=np.float64)
        
        # Set velocity value if it has been initialized - check correctness of minus sign
        self._velocity_weights = momentum * self._velocity_weights - lr * weights_grad
        self._velocity_bias = momentum * self._velocity_bias - lr * bias_grad

    def reset_velocity(self):
        self._velocity_weights = None
        self._velocity_biases = None
    
    # GETTER METHODS
    def get_weights(self):
        return self._weights
    
    def get_biases(self):
        return self._bias

    def get_z_values(self):
        return self._z_values

    # FUNCTIONAL METHODS
    def forward(self):
        # Get z values
        self._z_values = np.dot(self._weights, self._input)
        self._z_values = np.add(self._z_values, self._bias)

        # Apply activation function and get 
        self._output = self._activation_function(self._z_values.copy())

    def apply_activation_function_dx(self):
        return self._activation_function_dx(self._z_values)
    
    # Made with help from 
    def update_layer(self, weights_grad, bias_grad, lr, momentum=None, clip_value=1.0):
        # Apply clipping
        weights_grad_norm = np.linalg.norm(weights_grad)
        if weights_grad_norm > clip_value:
            weights_grad = weights_grad * (clip_value / weights_grad_norm)
            bias_grad = bias_grad * (clip_value / weights_grad_norm)

        # Initialize the values to change weights and bias by
        dW = None
        dB = None

        # Get values to update weights and bias by
        if not momentum is None:
            # Apply momentum
            self.update_velocity(weights_grad, bias_grad, momentum, lr)
            dW = self._velocity_weights
            dB = self._velocity_bias
        else:
            # Do not apply momentum
            dW = -lr * weights_grad
            dB = -lr * bias_grad

        # Update layer
        self._weights += dW
        self._bias += dB
    
    def contains_nan(self):
        return np.any(np.isnan(self._weights)) or np.any(np.isnan(self._bias))

    def __str__(self):
        to_ret = 'Weights:\n'
        to_ret += str(self._weights)
        to_ret += '\nBias:\n'
        to_ret += str(self._bias)
        to_ret += '\nNumber of inputs: '
        to_ret += str(self.num_inputs)
        to_ret += '\nNumber of outputs: '
        to_ret += str(self.num_outputs)
        to_ret += '\nActivation Function: '
        to_ret += self._activation_function.__name__
        return to_ret
    
    def __repr__(self):
        return self.__str__()



class NeuralNetwork(): 

    def __init__(
        self, 
        input_size:int=209,
        output_size:int=209,
        layers=None
    ):
        self._input = np.zeros(input_size, dtype=np.float64)
        self._layers = layers if not layers is None else []
        self._output = np.ones(output_size, dtype=np.float64)

    # ACTIVATION FUNCTIONS
    def relu(x: np.array):
        to_ret = []
        for val in x:
            if val > 0:
                to_ret.append(val)
            else:
                to_ret.append(0)
        return np.array(to_ret, dtype=np.float64)
    def relu_dx(x):
        to_ret = []
        for val in x:
            if val > 0:
                to_ret.append(1)
            else:
                to_ret.append(0)
        return np.array(to_ret, dtype=np.float64)
    def sigmoid(x: np.array):
        return 1/(1 + np.exp(-x))
    def sigmoid_dx(x: np.array):
        return NeuralNetwork.sigmoid(x) * (1 - NeuralNetwork.sigmoid(x))

    # GETTER METHODS
    def get_layer(self, layer_num):
        return self._layers[layer_num]
        
    def get_output(self):
        return self._output

    # SETTER METHODS
    def set_input(self, input: np.ndarray):
        if len(self._input) != len(input):
            raise ValueError("Error: length of provided input does not match input shape of network")
        self._input = input
    
    def append_layer(self, layer: Layer):
        # Check type of object provided
        if not isinstance(layer, Layer):
            raise TypeError("Error: please provide a Layer object or subtype")
        # Check dimensions are correct if it is a FeedForwardLayer
        if len(self._layers) > 0:
            if isinstance(layer, FeedForwardLayer) and isinstance(self._layers[-1], Layer) and self._layers[-1].num_outputs != layer.num_inputs:
                raise ValueError("Error: number of outputs of last layer does not match number of inputs of current layer")
        # Append layer
        self._layers.append(layer)

    # FUNCTIONAL METHODS
    def feed_forward(self):
        x = self._input.copy()
        for layer in self._layers:
            layer.set_input(x)
            layer.forward()
            x = layer.get_output()
        self._output = x

    # Made with assistance from http://neuralnetworksanddeeplearning.com/chap2.html
    def back_propogate(self, lr: float, error_prime: np.ndarray,  momentum=None, clip_score=1.0):
        # Get first delta value
        delta = error_prime * self._layers[-1].apply_activation_function_dx() # multiply gradient of cost function with derivative of activation function applied to z values
        
        # Do backprop for final layer
        bias_grad = delta
        weights_grad = np.outer(delta, self._layers[-2].get_output())
        self._layers[-1].update_layer(weights_grad, bias_grad, lr, momentum, clip_score)

        # Propogate through the other layers from the 2nd last hidden layer to the 1st hidden layer at index 0 of self._layers
        for l in range(len(self._layers) - 2, -1, -1):
            # Get each of the different layers
            curr_layer = self._layers[l]
            next_layer = self._layers[l+1]

            # Get s_l'(z_l) where s_l is activation function of layer l, the current layer
            act_fn_dx = curr_layer.apply_activation_function_dx()

            # Get delta_l = ((W_l+1)^T . delta_(l+1)) * s_l'(z_l) where W_l+1 is weights of next layer 
            delta = np.dot(next_layer.get_weights().transpose(), delta) * act_fn_dx

            # Get gradient for bias vector and weights matrix
            # Bias gradient is just delta
            bias_grad = delta
            # Weights gradient is delta . a_l-1 where a_l-1 is the activated output of the previous layer/input of current layer
            weights_grad = np.outer(delta, curr_layer.get_input()) # curr_layer.get_input() returns output of previous layer

            # Update current layer
            curr_layer.update_layer(weights_grad, bias_grad, lr, momentum, clip_score)

    def save_network(self, file_path: str):
        # Check that there are no NaN values in any of the layers
        for layer in self._layers:
            if layer.contains_nan():
                raise ValueError("Error: layer contains at least one NaN value")
            
        # Pickle network
        with open(file_path, 'wb') as f:
            pickle.dump(self, f)

    def load_network(file_path: str):
        # Load object
        to_ret = None
        with open(file_path, 'rb') as f:
            to_ret = pickle.load(f)

        # Check network is valid
        for i in range(len(to_ret._layers) - 1):
            curr_layer = to_ret.get_layer(i)
            next_layer = to_ret.get_layer(i + 1)
            if isinstance(curr_layer, FeedForwardLayer) and isinstance(next_layer,  FeedForwardLayer) and curr_layer.num_outputs != next_layer.num_inputs:
                raise ValueError("Error: invalid network provided")
            
        # Return network
        return to_ret
    
    def __str__(self):
        to_ret = '..................................................................'
        to_ret += '\n=================================================================='
        for i, layer in enumerate(self._layers):
            to_ret += f"\nLAYER {i + 1}:\n"
            to_ret += str(layer)
            to_ret += '\n=================================================================='
        to_ret += '\n..................................................................'
        return to_ret

    def __repr__(self):
        return self.__str__()


if __name__ == '__main__':
    # TEST CODE
    do_test = input("Would you like to test the neural network code? (y/n) ")
    if do_test.lower() == 'y':
        # NETWORK 1:
        # Create first layer and add it to network
        nn_1 = NeuralNetwork(input_size=2, output_size=2)
        nn_1_layer_1_weights = np.array([[0.15, 0.2], [0.25, 0.3]], dtype=np.float64)
        nn_1_layer_1_bias = np.array([0.35, 0.35], dtype=np.float64)
        nn_1_layer_1 = FeedForwardLayer(nn_1_layer_1_weights, nn_1_layer_1_bias, NeuralNetwork.sigmoid, NeuralNetwork.sigmoid_dx)
        nn_1.append_layer(nn_1_layer_1)
        # Create second layer and add it to network
        nn_1_layer_2_weights = np.array([[0.4, 0.45], [0.5, 0.55]], dtype=np.float64)
        nn_1_layer_2_bias = np.array([0.6, 0.6], dtype=np.float64)
        nn_1_layer_2 = FeedForwardLayer(nn_1_layer_2_weights, nn_1_layer_2_bias, NeuralNetwork.sigmoid, NeuralNetwork.sigmoid_dx)
        nn_1.append_layer(nn_1_layer_2)

        # CHECK FEEDFORWARD OF NETWORK
        print("Testing NETWORK 1:")
        print("TESTING FORWARD AND BACKWARDS PASS...")
        print(nn_1)
        print("FORWARD PASS:")
        print(f"Forward pass for [0.05, 0.1]:")
        nn_1.set_input(np.array([0.05, 0.1], dtype=np.float64))
        nn_1.feed_forward()
        output_1 = nn_1.get_output()
        print(output_1)
        print("Expected: [0.75136507, 0.772928465]")
        print()

        # CHECK BACKPROPOGATION OF NETWORK
        print("BACKWARD PASS:")
        print("Testing NETWORK 1:")
        target = np.array([0.01, 0.99], dtype=np.float64)
        print(f"Target of output:")
        print(target)
        error_prime = output_1 - target
        print(f"Error prime of output:")
        print(error_prime)
        nn_1.back_propogate(lr=0.5, error_prime=error_prime, clip_score=100.0)
        print("Updated values:")
        print(nn_1)
        print("Expected values:")
        print("Layer 1: [[0.149780716, 0.19956143], [0.24975114, 0.29950229]]")
        print("Layer 2: [[0.35891648, 0.408666186], [0.511301270, 0.561370121]]")
        print()

        # CHECK SAVING AND LOADING NETWORK
        print("TESTING SAVING AND LOADING OF NETWORK...")
        print("Testing saving of network...")
        nn_1.save_network('networks/test_networks/neural_network_1.pkl')
        print("Successfully saved network")
        print("Testing loading of network...")
        nn2 = NeuralNetwork.load_network('networks/test_networks/neural_network_1.pkl')
        print(nn2)
        print("Successfully loaded network")

        # CHECK SGD WITH MOMENTUM BACKPROP
        # TODO: test momentum



        # NETWORK 2:
        # Create first layer and add it to network
        nn_2 = NeuralNetwork(input_size=3, output_size=2)
        nn_2_layer_1_weights = np.array([
                              [0.2,  0.4,  0.1],
                              [0.5,  0.3,  0.7],
                              [0.6,  0.9,  0.2],
                              [0.8,  0.1,  0.5]
                             ])
        nn_2_layer_1_bias = np.array([0.1, 0.2, 0.3, 0.4])
        nn_2_layer_1 = FeedForwardLayer(nn_2_layer_1_weights, nn_2_layer_1_bias, NeuralNetwork.sigmoid, NeuralNetwork.sigmoid_dx)
        nn_2.append_layer(nn_2_layer_1)
        # Create second layer and add it to network
        nn_2_layer_2_weights = np.array([
                              [0.3,  0.7, 0.5, 0.9],
                              [0.8,  0.2, 0.6, 0.4]
                             ])
        nn_2_layer_2_bias = np.array([0.2, 0.5])
        nn_2_layer_2 = FeedForwardLayer(nn_2_layer_2_weights, nn_2_layer_2_bias, NeuralNetwork.sigmoid, NeuralNetwork.sigmoid_dx)
        nn_2.append_layer(nn_2_layer_2)

        # CHECK FEED FORWARD OF NETWORK
        print("\n\n\nTesting NETWORK 2:")
        print("TESTING FORWARD AND BACKWARDS PASS...")
        print(nn_2)
        print("FORWARD PASS:")
        print(f"Forward pass for [0.1, 0.5, 0.9]:")
        nn_2.set_input(np.array([0.1, 0.5, 0.9],  dtype=np.float64))
        nn_2.feed_forward()
        output_2 = nn_2.get_output()
        print(output_2)
        print("Expected: [0.87159201 0.86489426]")
        print()

        # CHECK BACKPROPOGATION OF NETWORK
        print("BACKWARD PASS:")
        print("Testing NETWORK 1:")
        target = np.array([0.0, 1.0], dtype=np.float64)
        print(f"Target of output:")
        print(target)
        error_prime = output_2 - target
        print(f"Error prime of output:")
        print(error_prime)
        nn_2.back_propogate(lr=0.5, error_prime=error_prime, clip_score=100.0)
        print("Updated values:")
        print(nn_2)
        print("Expected values:")
        print("""Layer 1: 
              [[0.19980057 0.39900284 0.09820511]
              [0.4993687  0.29684348 0.69431827]
              [0.59961186 0.8980593  0.19650674]
              [0.79919165 0.09595823 0.49272482]]""")
        print("""Layer 2:
              [[0.27068253 0.66405765 0.46443945 0.86453601]
              [0.80474481 0.20581699 0.6057552  0.40573957]]""")
        print()

        # CHECK SAVING AND LOADING NETWORK
        print("TESTING SAVING AND LOADING OF NETWORK...")
        print("Testing saving of network...")
        nn_2.save_network('networks/test_networks/neural_network_1.pkl')
        print("Successfully saved network")
        print("Testing loading of network...")
        nn3 = NeuralNetwork.load_network('networks/test_networks/neural_network_1.pkl')
        print(nn3)
        print("Successfully loaded network")

    # GENERATE A DUMMY NEURAL NETWORK TO USE FOR API TEMPORARILY
    generate_dummy_network = input("Would you like to generate a dummy neural network to use for API testing? (y/n) ")
    if generate_dummy_network.lower() == 'y':
        # generate dummy neural network that outputs 64 dimensional embedding
        nn = NeuralNetwork(
            input_size=202,
            output_size=202
        )
        # Layer 1
        layer_1_weights = np.random.rand(128, 202)
        layer_1_biases = np.random.rand(128)
        layer_1 = FeedForwardLayer(layer_1_weights, layer_1_biases, NeuralNetwork.relu)
        # Layer 2
        layer_2_weights = np.random.rand(64, 128)
        layer_2_biases = np.random.rand(64)
        layer_2 = FeedForwardLayer(layer_2_weights, layer_2_biases, NeuralNetwork.relu)
        # Add all layer to network
        nn.append_layer(layer_1)
        nn.append_layer(layer_2)
        # Save network
        nn.save_network('networks/test_networks/dummy_network.pkl')