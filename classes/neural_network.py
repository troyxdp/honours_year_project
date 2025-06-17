import numpy as np
import os


# TODO: 
# + add error checking for each of the methods


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

    def get_output(self):
        return self._output

class FeedForwardLayer(Layer):

    def __init__(
        self,
        weights:np.ndarray=None,
        biases:np.ndarray=None,
        activation_function=None,
    ):
        # Initialize superclass
        super().__init__()

        # Initialize forwarding properties
        self._weights = weights
        self._biases = biases
        self._activation_function = activation_function

        # Initialize forwarding results properties
        self._z_values = None

        # TODO: add backprop attributes like activation_function_dx

        # Set properties for dimensions
        self.num_inputs = None
        self.num_outputs = None
        if not self._weights is None:
            # X: 3 x 4 . y: 4 x 1 = 3 x 1 because X.q = y.p. We are trying to go from 4 inputs to 3 outputs in this example
            # X: 4 x 3 . y: 3 x 1 = 4 x 1 because X.q = y.p. We are trying to go from 3 inputs to 4 outputs in this example
            self.num_inputs = self._weights.shape[1] # number of inputs is number of columns
            self.num_outputs = self._weights.shape[0] # number of outputs is number of rows 

        # Create inputs and outputs
        if not self.num_inputs is None:
            self._input = np.zeros(self.num_inputs)
            self._output = np.zeros(self.num_outputs)

    def forward(self):
        # Get z values
        self._z_values = np.dot(self._weights, self._input)
        if not self._biases is None:
            self._z_values = np.add(self._z_values, self._biases)

        # Apply activation function and get 
        self._output = self._activation_function(self._z_values)

    def set_activation_function(self, act_fn):
        self._activation_function = act_fn

    def get_z_values(self):
        return self._z_values
    
    def update_layer(self, weights_grad, biases_grad=None):
        pass

    

class NeuralNetwork(): 

    def __init__(
        self, 
        input_size:int=209,
        output_size:int=209,
        layers=None
    ):
        self._input = np.zeros(input_size)
        self._layers = layers if not layers is None else []
        self._output = np.ones(output_size)
        self._output[:] = np.nan # Set output values to NaN

    def set_input(self, input: np.ndarray):
        if len(self._input) != len(input):
            raise ValueError("Error: length of provided input does not match input shape of network")
        self._input = input

    def feed_forward(self):
        x = self._input.copy()
        for layer in self._layers:
            layer.set_input(x)
            layer.forward()
            x = layer.get_output()
        self._output = x

    def back_propogate(self, lr: float, error_prime: np.ndarray,  momentum=None, clip_score=1.0):
        pass

    def get_output(self):
        return self._output
    
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

    def get_layer(self, layer_num):
        return self._layers[layer_num]

    def save_network(self, file_path: str):
        pass

    def load_network(self, file_path: str):
        pass