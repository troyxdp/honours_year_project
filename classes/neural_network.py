import numpy as np
import os

class NeuralNetwork(): 
    def __init__(self, num_neurons, activation_functions):
        self.num_neurons = num_neurons
        self.activation_functions = activation_functions