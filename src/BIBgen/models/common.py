from collections import OrderedDict

import torch
from torch import nn

class FourierEncoding(nn.Module):
        def __init__(self,
        dimension : int,
        initial_frequencies : torch.Tensor | None = None,
        learned : bool = True,
        log_frequency : bool = False,
        ):
        """
        Initializer for learned Fourier encoding module.
        Stores a vector of learnable frequencies,
        which are used to encode scalar input into vector Fourier representation.
        Inspired by positional encoding modules typically used in transformer models.

        Parameters
        ----------
        dimension : int
            Number of dimensions to represent scalar input. Must be even.
            Larger dimensions are more expressive of different length scales in the scalar input,
            but create more trainable parameters.
        initial_frequencies : torch.Tensor, optional
            Initial values to initiate learnable frequencies. Size must be half of `dimension`
            If not provided, `torch.arange(1, dimension // 2 + 1)` is used to initialize.
        learned : bool, optional
            To toggle to hard-coded frequencies. Currently not supported.

        Returns
        -------
        self : FourierEncoding
            torch module useable in neural networks

        Raises
        ------
        NotImplementedError
            If `learned` is toggled off
        ValueError
            If `dimension` is not even.
            If `len(initial_frequencies)` is not half of `dimension`.
        """
        super().__init__()
        if not learned:
            raise NotImplementedError("Unlearned encoding not supported.")
        if dimension % 2 != 0:
            raise ValueError("dimension must be even")
        half_dimension = dimension // 2

        if initial_frequencies is None:
            initial_frequencies = torch.arange(1, half_dimension+1, dtype=torch.float32)
        initial_frequencies = initial_frequencies.unsqueeze(0)

        self.log_frequency = log_frequency
        if self.log_frequency:
            self.frequency_table = nn.Parameter(data=torch.log(initial_frequencies))
        else:
            self.frequency_table = nn.Parameter(data=initial_frequencies)

    def forward(self, x : torch.Tensor) -> torch.Tensor:
        frequencies = torch.exp(self.frequency_table) if self.log_frequency else self.frequency_table
        thetas = x.unsqueeze(-1) @ frequencies
        sin_elems = torch.sin(thetas)
        cos_elems = torch.cos(thetas)
        return torch.cat((sin_elems, cos_elems), dim=-1)

class VarianceTower(nn.Module):
    SP_BETA = 1.0
    SP_THRESH = 20.0

    @staticmethod
    def softplus_inverse(y, beta, threshold):
        inv = (1.0 / beta) * torch.log(torch.expm1(beta * y))
        inv = torch.where(beta * y > threshold, y, inv)
        return inv

    def __init__(self, n_timesteps : int, initial_variances : torch.Tensor | None = None):
        """
        Initializer for learned variance tower module.
        Predicts the variance for all features (assumed to be the same)
        for every time step during the diffusion process.

        Parameters
        ----------
        n_timesteps : int
            The number of time steps in the diffusion process
        initial_variances : torch.Tensor, optional
            Initializing values for the variance lookup table with dimension (n_timesteps,).
            If not provided, random values between 0 and 1 are chosen.

        Returns
        -------
        self : VarianceTower
            torch module useable in neural networks

        Raises
        ------
        ValueError
            If `initial_variances` does not have size `max_steps`
        """
        super().__init__()
        if initial_variances is None:
            initial_variances = torch.rand(n_timesteps)
        if len(initial_variances) != n_timesteps:
            raise ValueError("initial_variances must have size n_timesteps")

        initial_values = self.softplus_inverse(initial_variances, self.SP_BETA, self.SP_THRESH)

        self.varhead_lookup_table = nn.Parameter(data=initial_values)
        self.varhead_activation = nn.Softplus(beta=self.SP_BETA, threshold=self.SP_THRESH)
        
    def forward(self, tau : torch.Tensor) -> torch.Tensor:
        variance_lookup = self.varhead_activation(self.varhead_lookup_table)
        return variance_lookup[tau - 1]

def make_mlp(input_size : int, output_size : int, hidden_layer_size : int, n_hidden_layers : int):
    assert n_hidden_layers >= 1
    
    stack = [("hidden0", nn.Linear(input_size, hidden_layer_size)), ("activation0", nn.ReLU())]
    for ilayer in range(1, n_hidden_layers):
        accumulator_stack.append(("hidden{}".format(ilayer), nn.Linear(hidden_layer_size, hidden_layer_size)), ("activation{}".format(ilayer), nn.ReLU()))

    accumulator_stack.append(("hidden{}".format(n_hidden_layers), nn.Linear(hidden_layer_size, output_size)))
    return nn.Sequential(OrderedDict(stack))
