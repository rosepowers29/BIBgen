from torch import nn
import numpy as np
from typing import TypeAlias

class DenseEncoder(nn.Module):
    def __init__(self, input_shape : tuple[int], latent_space_size : int, n_hidden_layers : int):
        super().__init__()

        self.input_shape = input_shape
        input_size = np.prod(input_shape)

        hidden_size_exp = 1
        while 2**hidden_size_exp < input_size:
            hidden_size_exp += 1
        hidden_size_exp -= 1

        self.hidden_size = 2**hidden_size_exp
        if self.hidden_size <= latent_space_size:
            self.hidden_size = round((input_size + latent_space_size) / 2)

        stack = [nn.Linear(input_size, self.hidden_size), nn.ReLU()]
        for ihidden in range(n_hidden_layers - 1):
            stack += [nn.Linear(self.hidden_size, self.hidden_size), nn.ReLU()]
        stack += [nn.Linear(self.hidden_size, latent_space_size), nn.ReLU(), nn.Linear(latent_space_size, self.hidden_size)]
        for ihidden in range(n_hidden_layers - 1):
            stack += [nn.Linear(self.hidden_size, self.hidden_size), nn.ReLU()]
        stack.append(nn.Linear(self.hidden_size, input_size))

        self.flatten = nn.Flatten()
        self.stack = nn.Sequential(*stack)

    def forward(self, x):
        x = self.flatten(x)
        reconstruction = self.stack(x)
        return reconstruction.view(self.input_shape)

# In case more types of encoders are added
Encoder : TypeAlias = DenseEncoder

class Decoder(nn.Module):
    def __init__(self, encoder : Encoder):
        # TODO
        