from collections import OrderedDict
from typing import Sequence

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

class PositionalEncoding(nn.Module):
    def __init__(self, dimension : int, scale : float):
        """
        Fixed, deterministic log-linear Fourier encoding ("positional encoding" in
        Tancik et al., "Fourier Features Let Networks Learn High Frequency Functions
        in Low Dimensional Domains", Sec. 6.1). Frequencies are axis-aligned and
        log-spaced, not learned:

            omega_j = 2*pi * scale ** (j / m),  j = 0, ..., m-1,  m = dimension // 2

        Parameters
        ----------
        dimension : int
            Number of dimensions to represent scalar input. Must be even.
        scale : float
            Frequency scale (sigma). Larger values reach higher frequencies but
            risk aliasing; tuned per problem via a held-out validation set.

        Returns
        -------
        self : PositionalEncoding
            torch module useable in neural networks

        Raises
        ------
        ValueError
            If `dimension` is not even.
        """
        super().__init__()
        if dimension % 2 != 0:
            raise ValueError("dimension must be even")
        half_dimension = dimension // 2

        powers = torch.arange(half_dimension, dtype=torch.float32) / half_dimension
        frequency_table = 2 * torch.pi * scale ** powers
        self.register_buffer("frequency_table", frequency_table.unsqueeze(0))

    def forward(self, x : torch.Tensor) -> torch.Tensor:
        thetas = x.unsqueeze(-1) @ self.frequency_table
        sin_elems = torch.sin(thetas)
        cos_elems = torch.cos(thetas)
        return torch.cat((sin_elems, cos_elems), dim=-1)

class GaussianFourierFeatures(nn.Module):
    def __init__(self, dimension : int, input_dim : int, scales : float | Sequence[float], seed : int | None = None):
        """
        Fixed random Fourier feature mapping jointly encoding a d-dimensional input
        (Tancik et al., Sec. 5-6, "Gaussian" mapping). Unlike `PositionalEncoding` /
        `FourierEncoding`, which each act on a single scalar coordinate, this mixes
        all input dimensions through a random frequency matrix B:

            B in R^{m x d}, B[j, i] ~ N(0, (2*pi*scales[i])**2),  m = dimension // 2
            gamma(v) = [cos(B v), sin(B v)]

        Parameters
        ----------
        dimension : int
            Number of dimensions of the output encoding. Must be even.
        input_dim : int
            Number of dimensions d of the input vector v being jointly encoded
            (e.g. 3 for phi, s, z).
        scales : float or Sequence[float]
            Per-input-dimension frequency scale (sigma). A scalar is broadcast to
            every input dimension; a sequence of length `input_dim` gives an
            independent sigma per dimension (e.g. one each for phi, s, z),
            sampling B anisotropically.
        seed : int, optional
            If given, frequencies are sampled from a local `torch.Generator` seeded
            with this value, so the draw is reproducible without disturbing the
            global RNG (e.g. data shuffling) elsewhere in training.

        Returns
        -------
        self : GaussianFourierFeatures
            torch module useable in neural networks

        Raises
        ------
        ValueError
            If `dimension` is not even, or `scales` is a sequence whose length
            does not match `input_dim`.
        """
        super().__init__()
        if dimension % 2 != 0:
            raise ValueError("dimension must be even")
        half_dimension = dimension // 2

        scales = torch.as_tensor(scales, dtype=torch.float32)
        if scales.dim() == 0:
            scales = scales.repeat(input_dim)
        elif scales.numel() != input_dim:
            raise ValueError("scales must be a scalar or have length input_dim")

        generator = torch.Generator().manual_seed(seed) if seed is not None else None
        raw = torch.randn(half_dimension, input_dim, generator=generator)
        frequency_table = 2 * torch.pi * raw * scales
        self.register_buffer("frequency_table", frequency_table)

    def forward(self, x : torch.Tensor) -> torch.Tensor:
        thetas = x @ self.frequency_table.T
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
