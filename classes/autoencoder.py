import numpy as np
import torch

class Autoencoder(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.layer_1 = torch.nn.Linear(202, 165)
        self.layer_2 = torch.nn.Linear(165, 128)
        self.layer_3 = torch.nn.Linear(128, 64)
        self.layer_4 = torch.nn.Linear(64, 128)
        self.layer_5 = torch.nn.Linear(128, 165)
        self.layer_6 = torch.nn.Linear(165, 202)

    def forward(self, x):
        # Check valid dimensions are given
        if len(x) != 202:
            raise ValueError("Error: invalid input provided. Length of input wrong")
        
        # Feed through network
        output = torch.nn.functional.relu(self.layer_1(x))
        output = torch.nn.functional.relu(self.layer_2(output))
        output = torch.nn.functional.relu(self.layer_3(output))
        output = torch.nn.functional.relu(self.layer_4(output))
        output = torch.nn.functional.relu(self.layer_5(output))
        output = torch.nn.functional.relu(self.layer_6(output))

        # Return result
        return output
    
    def encode(self, x):
        # Check valid dimensions are given
        if len(x) != 202:
            raise ValueError("Error: invalid input provided. Length of input wrong")

        # Feed through encoder section
        embedding = torch.nn.functional.relu(self.layer_1(x))
        embedding = torch.nn.functional.relu(self.layer_2(embedding))
        embedding = torch.nn.functional.relu(self.layer_3(embedding))

        # Return result
        return embedding