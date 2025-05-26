import numpy as np
import os

class NeuralNetwork(): 
    def __init__(self, num_neurons, activation_functions):
        # Check for invalid initialization parameters
        if (len(num_neurons) - 1) != len(activation_functions):
            raise Exception("Error: incorrect number of activation functions given")

        # Check network has at least one hidden layer
        if (len(num_neurons) < 3):
            raise Exception("Error: network must have at least one hidden layer")

        # z values
        self.z_values = []

        # initialize layers    
        self.layers = []
        for i in num_neurons:
            self.layers.append(np.zeros(i))
            self.z_values.append(np.zeros(i))        

        # initialize weights
        self.weights = []
        for i in range(1, len(num_neurons)):
            # Initialize with He initialization
            self.weights.append(np.random.normal(0, np.sqrt(2/ num_neurons[i-1]), (num_neurons[i], num_neurons[i-1])))

        # Initialize biases
        self.biases = []
        for i in range(1, len(num_neurons)):
            self.biases.append(np.random.rand(num_neurons[i])/100)

        # Save the number of neurons in each layer
        self.num_neurons = np.array(num_neurons)

        # Initialize activation functions and their derivatives
        self.activation_functions = []
        self.activation_function_derivatives = []
        for activation_function in activation_functions:
            if activation_function == 'sigmoid':
                self.activation_functions.append(self.sigmoid)
                self.activation_function_derivatives.append(self.sigmoid_dx)
            elif activation_function == 'relu':
                self.activation_functions.append(self.relu)
                self.activation_function_derivatives.append(self.relu_dx)
            else:
                self.activation_functions.append(self.linear)
                self.activation_function_derivatives.append(self.linear_dx)

    # ACTIVATION FUNCTIONS
    def sigmoid(self, x):
        return 1/(1 + np.exp(-x))

    def sigmoid_dx(self, x):
        return self.sigmoid(x) * (1 - self.sigmoid(x))

    def relu(self, x):
        to_ret = []
        for val in x:
            if val > 0:
                to_ret.append(val)
            else:
                to_ret.append(0)
        return np.array(to_ret)

    def relu_dx(self, x):
        to_ret = []
        for val in x:
            if val > 0:
                to_ret.append(1)
            else:
                to_ret.append(0)
        return np.array(to_ret)

    def linear(self, x):
        return x

    def linear_dx(self, x):
        return 1

    # SETTER METHODS
    def set_weights(self, weight_matrices):
        # Check number of weight matrices is correct
        if len(weight_matrices) != len(self.weights):
            raise Exception
        # Check dimensions of weight matrices are correct
        for i in range(len(weight_matrices)):
            if not weight_matrices[i].shape == self.weights[i].shape:
                raise Exception

        # Copy across values
        for i in range(len(weight_matrices)):
            self.weights[i] = weight_matrices[i].copy()

    def set_biases(self, bias_vectors):
        # Check number of bias vectors is correct
        if len(bias_vectors) != len(self.biases):
            raise Exception
        # Check dimensions of bias vectors
        for i in range(len(self.biases)):
            if not len(self.biases[i]) == len(bias_vectors[i]):
                raise Exception
        
        # Copy values across
        for i in range(len(self.biases)):
            self.biases[i] = bias_vectors[i].copy()

    # METHODS RELATED TO FEED FORWARD OF INPUT
    def feed_forward(self, nn_input):
        # Check length of input is correct
        if len(self.layers[0]) != len(nn_input):
            raise Exception

        # Copy input across to input layer
        for i in range(len(self.layers[0])):
            self.layers[0][i] = nn_input[i]

        # Feed forward through the layers
        self.layers[0] = np.array(self.layers[0])
        for i in range(1, len(self.layers)):
            layer = self.layers[i-1]
            weight_values = self.weights[i-1]
            bias_values = self.biases[i-1]
            z = np.dot(weight_values, layer)
            for j in range(len(self.z_values[i])):
                self.z_values[i][j] = z[j] + bias_values[j]
            self.layers[i] = self.activation_functions[i-1](self.z_values[i].copy())

        # Return output
        return np.array(self.layers[-1])

    def get_output(self):
        return self.layers[-1]

    # METHODS RELATED TO UPDATING NETWORK
    def back_propogate(self, delta):
        # Get backpropogation values for each weights matrix and bias vector
        grad_w = [np.zeros(weight_matrix.shape) for weight_matrix in self.weights]
        grad_b = [np.zeros(bias_vector.shape) for bias_vector in self.biases]

        # Get initial backpropogation values
        grad_b[-1] = delta.copy()
        grad_w[-1] = np.outer(delta, self.layers[-2])

        # Backpropogate throughout layers
        d = delta.copy()
        for l in range(2, len(self.layers)): # iterate backwards through layers
            # s_l'(z_l) where s_l is activation function of layer l
            act_fn_dx = self.activation_function_derivatives[-l](self.z_values[-l]) 
            # delta_l = ((W_l+1)^T . delta_(l+1)) * s_l'(z_l) where W_l+1 is weights of next layer 
            d = np.dot(self.weights[-l+1].transpose(), d) * act_fn_dx 
            # gradient for biases = delta_l
            grad_b[-l] = d 
            # gradient for weights = delta_l . (a_l-1)^T where a_l-1 is the activated outputs of the previous layer
            grad_w[-l] = np.outer(d, self.layers[-l-1]) 
        # Return gradients
        return grad_b, grad_w

    def update_network(self, lr, error_prime, use_gradient_clipping=True, clip_value=1.0):
        # delta = Grad(Cost(x)) * activation_function_prime(z_L)
        delta = error_prime * self.activation_function_derivatives[-1](self.z_values[-1])
        # Get error values
        grad_b, grad_w = self.back_propogate(delta)

        # Update weights
        for i in range(len(self.weights)): # iterate through weight matrices and bias vectors
            # Get gradients
            dW = grad_w[i]
            dB = grad_b[i]

            # Perform  norm-based gradient clipping
            dW_norm = np.linalg.norm(dW)
            if dW_norm > clip_value:
                dW = dW * (clip_value / dW_norm)
                dB = dB * (clip_value / dW_norm)

            # Update weights
            for j in range(len(self.weights[i])): # iterate through rows of weight matrices and biases
                for k in range(len(self.weights[i][j])): # iterate through columns of weight matrices
                    self.weights[i][j][k] -= lr * dW[j][k]
                self.biases[i][j] -= lr * dB[j]

    # UTILITY METHODS
    def __str__(self):
        to_ret = ''
        for i, weight_matrix in enumerate(self.weights):
            to_ret += f'Layer {i}:\n'
            to_ret += f'Weights:\n'
            to_ret += str(weight_matrix) + '\n'
            to_ret += 'Biases:\n'
            to_ret += str(self.biases[i])
            to_ret += '\n\n'
        return to_ret[:-2]

    def reset_layers(self):
        for layer in self.layers:
            for i in range(len(layer)):
                layer[i] = 0

    # METHODS FOR SAVING AND LOADING NETWORK TO AND FROM FILES
    def save_network(self, dir_path):
        # Save number of neurons
        np.save(os.path.join(dir_path, 'num_neurons.npy'), np.array(self.num_neurons))

        # Save weights and biases
        for i in range(len(self.weights)):
            # Check for NaN values
            if np.isnan(self.weights[i]).any() or np.isnan(self.biases[i]).any():
                raise Exception("Error: trying to save NaN values")
            # Save weights and biases for layer i    
            np.save(os.path.join(dir_path, f'weights_{i}.npy'), np.array(self.weights[i]))
            np.save(os.path.join(dir_path, f'biases_{i}.npy'), np.array(self.biases[i]))

        # Save activation functions
        with open(os.path.join(dir_path, 'activation_functions.txt'), 'w') as f:
            to_write = ''
            for activation_function in self.activation_functions:
                if activation_function.__name__ == 'sigmoid':
                    to_write += 'sigmoid\n'
                elif activation_function.__name__ == 'relu':
                    to_write += 'relu\n'
                else:
                    to_write += 'linear\n'
            f.write(to_write[:-1])

    def load_network(self, dir_path):
        # Get number of neurons in each layer
        self.num_neurons = np.load(os.path.join(dir_path, 'num_neurons.npy'))

        # Load weights and biases
        weight_matrices = []
        bias_vectors = []
        for i in range(0, len(self.num_neurons) - 1):
            # Load weights and biases for a specific layer
            weights = np.load(os.path.join(dir_path, f'weights_{i}.npy'), allow_pickle=True)
            biases = np.load(os.path.join(dir_path, f'biases_{i}.npy'), allow_pickle=True)

            # Check dimensions given for weights matrix
            if not len(weights) == self.num_neurons[i+1] or not len(weights[0]) == self.num_neurons[i]:
                raise Exception("Error: invalid network provided -  dimensions of weight matrices do not match number of neurons given for each layer")
            # Check dimensions given for bias vectors
            if not len(biases) == self.num_neurons[i+1]:
                raise Exception("Error: invalid network provided - dimensions of bias vectors do not match number of neurons given for each layer")
            # Check if any of the values are NaNs
            if np.isnan(weights).any() or np.isnan(biases).any():
                raise Exception(f"NaNs detected in weights/biases at layer {i}")

            # Store them if valid
            weight_matrices.append(weights)
            bias_vectors.append(biases)

        # Load weight_matrices and bias_vectors values into object
        self.set_weights(weight_matrices)
        self.set_biases(bias_vectors)

        # Load activation functions
        self.activation_functions = []
        self.activation_function_derivatives = []
        with open(os.path.join(dir_path, 'activation_functions.txt'), 'r') as f:
            activation_functions = f.readlines()
            for activation_function in activation_functions:
                fn = activation_function.strip()
                if fn == 'sigmoid':
                    self.activation_functions.append(self.sigmoid)
                    self.activation_function_derivatives.append(self.sigmoid_dx)
                elif fn == 'relu':
                    self.activation_functions.append(self.relu)
                    self.activation_function_derivatives.append(self.relu_dx)
                else:
                    self.activation_functions.append(self.linear)
                    self.activation_function_derivatives.append(self.linear_dx)



# Test script
if __name__ == '__main__':
    # nn = NeuralNetwork((43, 128, 128, 7), ('relu', 'relu', 'linear'))
    # for dir_name in sorted(os.listdir('/home/troyxdp/Documents/University Work/Artificial Intelligence/Project/results/training_round_2')):
    #     print(f'{dir_name}')
    #     path = os.path.join('/home/troyxdp/Documents/University Work/Artificial Intelligence/Project/results/training_round_2', dir_name)
    #     if not os.path.isdir(path):
    #         continue
    #     nn.load_network(path)
    #     print(nn)
    #     print()

    # Create a first neural network
    print("==================================================================================")
    print("NETWORK 1:")
    nn = NeuralNetwork((2, 2, 2), ('sigmoid', 'sigmoid'))
    nn.set_weights([np.array([[0.15, 0.2], [0.25, 0.3]]), np.array([[0.4, 0.45], [0.5, 0.55]])])
    nn.set_biases([np.array([0.35, 0.35]), np.array([0.6, 0.6])])

    # Feedforward
    print("Network Output:")
    print(output:=nn.feed_forward([0.05, 0.1]))

    # Update network using backpropogation
    target = np.array([0.01, 0.99])
    # error = (output - target) * sigmoid_dx(nn.z_values[-1])
    error = (output - target)
    print("\nError of output:")
    print(error)
    nn.update_network(0.5, error)
    print("\nFinal network:")
    print(nn)
    print("==================================================================================")



    # Create a second neural network
    print("\n\n\n==================================================================================")
    print("NETWORK 2:")
    nn = NeuralNetwork((2, 2, 1), ('sigmoid', 'sigmoid'))
    nn.set_weights([np.array([[0.15, 0.2], [0.25, 0.3]]), np.array([[0.4, 0.45]])])
    nn.set_biases([np.array([0.35, 0.35]), np.array([0.6])])

    # Test feedforward
    print("Network Output:")
    print(output:=nn.feed_forward([0.05, 0.1]))

    # Test backpropogation
    target = np.array([0.01])
    error = (output - target)
    print("\nError of output:")
    print(error)
    nn.update_network(0.5, error)
    print("\nFinal network:")
    print(nn)
    print("==================================================================================")



    # Create a third neural network
    print("\n\n\n==================================================================================")
    print("NETWORK 3:")
    nn = NeuralNetwork((3, 4, 2), ('sigmoid', 'sigmoid'))
    nn.set_weights([np.array([
                              [0.2,  0.4,  0.1],
                              [0.5,  0.3,  0.7],
                              [0.6,  0.9,  0.2],
                              [0.8,  0.1,  0.5]
                             ]), 
                    np.array([
                              [0.3,  0.7, 0.5, 0.9],
                              [0.8,  0.2, 0.6, 0.4]
                             ]),
                   ])
    nn.set_biases([np.array([0.1, 0.2, 0.3, 0.4]), np.array([0.2, 0.5])])
    
    # Test feedforward
    print("Network Output:")
    print(output:=nn.feed_forward(np.array([0.1, 0.5, 0.9])))

    # Test backpropogation
    target = np.array([0.0, 1.0])
    error = (output - target)
    print("\nError of output:")
    print(error)
    nn.update_network(0.5, error)
    print("\nFinal network:")
    print(nn)
    print("==================================================================================")



    print("\n\n\n==================================================================================")
    print("NETWORK 4:")
    nn = NeuralNetwork((3, 4, 2), ('relu', 'sigmoid'))
    nn.set_weights([np.array([
                              [0.2,  0.4,  0.1],
                              [0.5,  0.3,  0.7],
                              [0.6,  0.9,  0.2],
                              [0.8,  0.1,  0.5]
                             ]), 
                    np.array([
                              [0.3,  0.7, 0.5, 0.9],
                              [0.8,  0.2, 0.6, 0.4]
                             ]),
                   ])
    nn.set_biases([np.array([0.1, 0.2, 0.3, 0.4]), np.array([0.15, 0.5])])
    
    # Test feedforward
    print("Network Output:")
    print(output:=nn.feed_forward(np.array([0.9, 0.1, 0.8])))

    # Test backpropogation
    target = np.array([0.2, 0.6])
    error = (output - target)
    print("\nError of output:")
    print(error)
    nn.update_network(0.1, error)
    print("\nFinal network:")
    print(nn)
    print("==================================================================================")



    print("\n\n\n==================================================================================")
    print("Testing saving NETWORK 4...")
    # Testing saving of network
    nn.save_network(os.path.join(os.getcwd(), 'architecture'))

    # Test loading of network
    print("Testing loading NETWORK 4 from saved files...\n")
    nn.load_network(os.path.join(os.getcwd(), 'architecture'))
    print(nn)
    print("==================================================================================")

    # Get error
    # Delta_L = Grad(CostFn)(target) * OutputActivationFunctionDerivative, which in our case is: 
    # error = np.zeros(len(self.layers[-1]))
    # error[target_a] = (self.layers[-1][target_a] - target_q)
    # error = error * self.activation_function_derivatives[-1](self.z_values[-1]) # * 1 since output activation function f(x) = x