import numpy as np
import os


# TODO: add error checking for each of the methods
# TODO: implement pretty printing of neural network and each layer


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
            self._velocity_weights = np.zeros_like(weights_grad)
            self._velocity_bias = np.zeros_like(bias_grad)
        
        # Set velocity value if it has been initialized
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
        self._output = self._activation_function(self._z_values)

    def apply_activation_function_dx(self):
        return self._activation_function_dx(self._z_values)
    
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
    

class NeuralNetwork(): 

    # TODO: add model summary __str__ method
    # TODO: add saving and loading
    # TODO: do backprop

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

    def back_propogate(self, lr: float, error_prime: np.ndarray,  momentum=None, clip_score=1.0):
        # Get first delta value
        delta = self._layers[-1].get_output() * error_prime
        
        # Do backprop for final layer
        bias_grad = delta
        weights_grad = np.outer(delta, self._layers[-2].get_weights())
        self._layers[-1].update_layer(weights_grad, bias_grad, lr, momentum, clip_score)

        # Propogate through the rest of the layers
        d = delta.copy()
        for l in range(2, len(self._layers)):
            # Get s_l'(z_l) where s_l is activation function of layer l
            act_fn_dx = self._layers[-l].apply_activation_function_dx()

            # Get delta_l = ((W_l+1)^T . delta_(l+1)) * s_l'(z_l) where W_l+1 is weights of next layer 
            d = np.dot(self._layers[-l+1].get_weights().transpose(), d) * act_fn_dx

            # Get gradient for bias vector and weights matrix
            bias_grad = d
            weights_grad = np.outer(d, self._layers[-l+1].get_output())

            # Update layer
            self._layers[-l].update_layer(weights_grad, bias_grad, lr, momentum, clip_score)

    def save_network(self, file_path: str):
        pass

    def load_network(self, file_path: str):
        pass