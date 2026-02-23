from typing import Callable, Collection
from collections import OrderedDict

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
        initial_frequencies = initial_frequencies.unsqueeze(0) # (1, dimension / 2)

        self.frequency_table = nn.Parameter(data=initial_frequencies)

    def forward(self, x : torch.Tensor) -> torch.Tensor:
        """
        Forward pass of fourier encoding. Input `x` is multiplied by every frequency,
        which then both `sin` and `cos` are taken of to add to the encoding vector.
        Note that this module should be called directly with __call__ when incorporated in neural networks.

        Parameters
        ----------
        x : torch.Tensor
            Tensor of scalar inputs to be encoded in fourier vector representation,
            with dimension `(n_batch, n_members)` or `(n_members,)`

        Returns
        -------
        x_fourier : torch.Tensor
            Tensor of fourier vector representation with dimension `(n_batch, n_members, dimension)` or `(n_members, dimension)`
        """
        thetas = x.unsqueeze(-1) @ self.frequency_table
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

class EquivariantLayer(nn.Module):
    def __init__(self, input_size : int, output_size : int):
        """
        Initializer for permutation equivariant layer as described
        in https://arxiv.org/abs/1703.06114.

        Parameters
        ----------
        input_size : int
            Size of each input set member.
            Note this is different from the number of memhers.
        output_size : int
            Soze of each output member.

        Returns
        -------
        self : EquivariantLayer
            torch module useable in neural networks
        """
        super().__init__()

        self.lambda_mat = nn.Parameter(torch.empty(output_size, input_size))
        self.gamma_mat = nn.Parameter(torch.empty(output_size, input_size))
        self.bias_vec = nn.Parameter(torch.zeros(output_size))

        nn.init.kaiming_uniform_(self.lambda_mat, a=0.0, nonlinearity="relu")
        nn.init.kaiming_uniform_(self.gamma_mat,  a=0.0, nonlinearity="relu")

    def forward(self, input_set : torch.Tensor):
        r"""
        Forward pass that computes
        $(\lambda I + \gamma (1 1^\top)) \vec{x} + \vec{b}$

        Parameters
        ----------
        input_set : torch.Tensor
            Input set with shape (n_batch, n_members, input_size) or (n_members, input_size)

        Returns
        -------
        output_set : torch.Tensor
            Output set with shape (n_batch, n_members, output_size) or (n_members, output_size)
        """
        n_members = input_set.size()[-2]
        x = input_set.unsqueeze(-1) # (n_batch, n_members, input_size, 1)

        self_term = input_set @ self.lambda_mat.T
        x_mean = input_set.mean(dim=-2)
        output_sizes = [-1] * input_set.dim()
        output_sizes[-2] = n_members
        interaction = (x_mean @ self.gamma_mat.T).unsqueeze(-2).expand(*output_sizes)
        
        return self_term + interaction + self.bias_vec

class EquivariantDenoiser(nn.Module):
    def __init__(self,
        n_timesteps : int,
        tau_encoding_dimension : int,
        position_encoding_dimension : int,
        hidden_layer_size : int,
        n_hidden_layers : int,
        nhits_normalization : int = 10_000,
        predict_variances : bool = False
    ):
        """
        Denoising model using a deep equivariant tower for prediction.

        Parameters
        ----------
        n_timesteps : int
            Number of diffusion time steps
        tau_encoding_dimension : int
            Number of dimensions to encode diffusion time
        position_encoding_dimension : int
            Number of dimensions to encode each spatial dimention
        hidden_layer_size : int
            Size of hidden equivariant layers in prediction tower
        n_hidden_layers : int
            Number of hidden layers in prediction towers
        nhits_normalization : int
            Number to divide the number of hits before it goes in as a feature
        betas : torch.Tensor, optional
            Diffusion schedule with shape (n_timesteps,).
            Used to initiate the variance tower.
        """
        super().__init__()

        self.pos1_encoding = FourierEncoding(position_encoding_dimension)
        self.pos2_encoding = FourierEncoding(position_encoding_dimension)
        self.pos3_encoding = FourierEncoding(position_encoding_dimension)
        self.tau_encoding = FourierEncoding(tau_encoding_dimension)
        encoding_size = tau_encoding_dimension + 1 + 1 + 3 * position_encoding_dimension
        self.n_timesteps = n_timesteps
        self.nhits_norm = nhits_normalization

        equivariant_layers = [
            ("hidden0", EquivariantLayer(encoding_size, hidden_layer_size)),
            ("activation0", nn.ReLU())
        ]
        for ihidden in range(1, n_hidden_layers):
            equivariant_layers += [
                ("hidden{}".format(ihidden), EquivariantLayer(hidden_layer_size, hidden_layer_size)),
                ("activation{}".format(ihidden), nn.ReLU())
            ]

        self.predict_variances = predict_variances
        output_size = 4 * 2 if self.predict_variances else 4
        equivariant_layers.append(("prediction_output", EquivariantLayer(hidden_layer_size, output_size)))

        self.prediction_tower = nn.Sequential(OrderedDict(equivariant_layers))

    def forward(self, input_set : torch.Tensor, tau : torch.Tensor):
        """
        Forward pass that predicts `tau` state of `input_set`,
        and the associated variance of the prediction.

        Parameters
        ----------
        input_set : torch.Tensor
            Input set at 'tau + 1' state with shape (n_batch, n_members, 4) or (n_members, 4) with features:
            (Edepm, x1, x2, x3).
        tau : torch.Tensor
            Diffusion time step of `input_set` with shape (n_batch,) if batched.

        Returns
        -------
        prediction : torch.Tensor
            Prediction of the `tau` state also with shape (n_batch, n_members, 4) or (n_members, 4)
        variance : float
            Variance of all elements of the prediction if requested.
        """
        nhits = input_set.shape[-2]

        tau_encoded = self.tau_encoding(tau / self.n_timesteps) # (n_batch, tau_encoding_dimension)
        tau_encoded = tau_encoded.unsqueeze(-2).expand(*tau_encoded.shape[:-1], nhits, -1) # (n_batch, n_members, tau_encoding_dimension)
    
        pos1_encoded = self.pos1_encoding(input_set[...,1]) # (n_batch, n_members, positional_encoding_dimension)
        pos2_encoded = self.pos2_encoding(input_set[...,2]) # (n_batch, n_members, positional_encoding_dimension)
        pos3_encoded = self.pos3_encoding(input_set[...,3]) # (n_batch, n_members, positional_encoding_dimension)

        nhits_feature = input_set.new_full((*input_set.shape[:-1], 1), nhits / self.nhits_norm) # (n_batch, n_members, 1)

        encoded_set = torch.cat((tau_encoded, input_set[...,0:1], nhits_feature, pos1_encoded, pos2_encoded, pos3_encoded), axis=-1)
        out = self.prediction_tower(encoded_set)

        if not self.predict_variances:
            return out

        prediction = out[...,:4]
        variances = torch.nn.functional.softplus(out[..., 4:])
        return prediction, variances