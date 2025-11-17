from typing import Callable, Collection

import numpy as np
import torch
from torch import nn

class FourierEncoding(nn.Module):
    def __init__(self, dimension : int, initial_frequencies : torch.Tensor | None = None, learned : bool = True):
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
            If not provided, `0.5**torch.linspace(1, 16, dimension)` is used to initialize.
        learned : bool
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
            initial_frequencies = 0.5**torch.linspace(1, 16, half_dimension)
        if len(initial_frequencies) != half_dimension:
            raise ValueError("Size of initial_frequencies is not half of dimension")

        # Inverse sigmoid
        initial_values = - torch.log((1 / initial_frequencies) - 1)

        self.fourier_table = nn.Parameter(data=initial_values)
        self.fourier_activation = nn.Sigmoid()

    def forward(self, x : torch.Tensor) -> torch.Tensor:
        """
        Forward pass of fourier encoding. Input `x` is multiplied by every frequency,
        which then both `sin` and `cos` are taken of to add to the encoding vector.
        Note that this module should be called directly with __call__ when incorporated in neural networks.

        Parameters
        ----------
        x : torch.Tensor
            Tensor of scalar inputs to be encoded in fourier vector representation,
            with dimension `(n_batch,)`

        Returns
        -------
        x_fourier : torch.Tensor
            Tensor of fourier vector representation with dimension `(n_batch, dimension)`
        """
        frequencies = 2 * np.pi * self.fourier_activation(self.fourier_table)
        thetas = torch.outer(x, frequencies)
        sin_elems = torch.sin(thetas)
        cos_elems = torch.cos(thetas)
        return torch.cat((sin_elems, cos_elems), dim=1)

class VarianceHead(nn.Module):
    SP_BETA = 1.0
    SP_THRESH = 20.0

    @staticmethod
    def softplus_inverse(y, beta, threshold):
        inv = (1.0 / beta) * torch.log(torch.expm1(beta * y))
        inv = torch.where(beta * y > threshold, y, inv)
        return inv

    def __init__(self, n_timesteps : int, initial_variances : torch.Tensor | None = None):
        """
        Initializer for learned variance head module.
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
        self : VarianceHead
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
        return variance_lookup[tau]
