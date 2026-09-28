from collections import OrderedDict
from typing import Sequence

import torch
from torch import nn

from BIBgen.models import common

class EquivariantLayer(nn.Module):
    def __init__(self, input_size : int, output_size : int, initialize_zero : bool = False):
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

        if not initialize_zero:
            nn.init.kaiming_uniform_(self.lambda_mat, a=0.0, nonlinearity="relu")
            nn.init.kaiming_uniform_(self.gamma_mat,  a=0.0, nonlinearity="relu")
        else:
            nn.init.zeros_(self.lambda_mat)
            nn.init.zeros_(self.gamma_mat)

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
    @staticmethod
    def _axis_scales(position_encoding_scale):
        """Normalize position_encoding_scale into an explicit (sigma_phi, sigma_s, sigma_z) triple."""
        if position_encoding_scale is None:
            raise ValueError("position_encoding_scale is required for position_encoding_kind != 'learned'")
        if isinstance(position_encoding_scale, (int, float)):
            return (position_encoding_scale,) * 3
        sigma_phi, sigma_s, sigma_z = position_encoding_scale
        return (sigma_phi, sigma_s, sigma_z)

    def get_layer(self, insize, outsize, initialize_zero=False):
        if self.disable_interactions:
            return nn.Linear(insize, outsize)
        return EquivariantLayer(insize, outsize, initialize_zero)

    def __init__(self,
        n_timesteps : int,
        tau_encoding_dimension : int,
        position_encoding_dimension : int,
        hidden_layer_size : int,
        n_hidden_layers : int,
        nhits_normalization : int = 9_000,
        predict_variances : bool = False,
        disable_interactions : bool = False,
        use_position_encoding : bool = False,   # default flipped
        position_encoding_kind : str = "learned",
        position_encoding_scale : float | Sequence[float] | None = None,
        position_encoding_seed : int | None = None,
        log_frequency : bool = False,
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
            Number of dimensions to encode each spatial dimention.
            For `position_encoding_kind="gaussian"` this is the width of the single
            joint (phi, s, z) encoder; for "learned"/"positional" it is the width of
            each of the three independent per-axis encoders.
        hidden_layer_size : int
            Size of hidden equivariant layers in prediction tower
        n_hidden_layers : int
            Number of hidden layers in prediction towers
        nhits_normalization : int
            Number to divide the number of hits before it goes in as a feature
        use_position_encoding : bool
            Whether to feed a Fourier-encoded (phi, s, z) into the network in
            addition to the raw coordinates.
        position_encoding_kind : str
            One of "learned" (default; gradient-trained frequencies via
            `common.FourierEncoding`, applied independently per axis -- kept as a
            baseline), "positional" (fixed, deterministic log-linear frequencies
            via `common.PositionalEncoding`, applied independently per axis), or
            "gaussian" (fixed random frequencies via `common.GaussianFourierFeatures`,
            jointly mixing phi, s, z through a single encoder). Only used when
            `use_position_encoding=True`.
        position_encoding_scale : float or Sequence[float], optional
            Frequency scale (sigma) for "positional"/"gaussian" kinds. A scalar is
            shared across phi, s, z; a length-3 sequence gives independent
            (sigma_phi, sigma_s, sigma_z). Required (and otherwise ignored) unless
            `position_encoding_kind="learned"`.
        position_encoding_seed : int, optional
            Only used when `position_encoding_kind="gaussian"`: seeds the random
            draw of the frequency matrix B (see `common.GaussianFourierFeatures`)
            so it's reproducible across runs, instead of a fresh random draw every
            time. Ignored for "learned"/"positional", which have no randomness to
            seed.
        log_frequency : bool
            Only used when `position_encoding_kind="learned"`: reparametrize the
            learned frequencies in log-space (see `common.FourierEncoding`).
        betas : torch.Tensor, optional
            Diffusion schedule with shape (n_timesteps,).
            Used to initiate the variance tower.
        """
        super().__init__()

        self.n_timesteps = n_timesteps
        self.nhits_norm = nhits_normalization
        self.disable_interactions = disable_interactions
        self.use_position_encoding = use_position_encoding
        self.position_encoding_kind = position_encoding_kind

        if self.use_position_encoding:
            if position_encoding_kind == "learned":
                self.pos1_encoding = common.FourierEncoding(position_encoding_dimension, log_frequency=log_frequency)
                self.pos2_encoding = common.FourierEncoding(position_encoding_dimension, log_frequency=log_frequency)
                self.pos3_encoding = common.FourierEncoding(position_encoding_dimension, log_frequency=log_frequency)
                pos_size = 3 * position_encoding_dimension
            elif position_encoding_kind == "positional":
                sigma_phi, sigma_s, sigma_z = self._axis_scales(position_encoding_scale)
                self.pos1_encoding = common.PositionalEncoding(position_encoding_dimension, scale=sigma_phi)
                self.pos2_encoding = common.PositionalEncoding(position_encoding_dimension, scale=sigma_s)
                self.pos3_encoding = common.PositionalEncoding(position_encoding_dimension, scale=sigma_z)
                pos_size = 3 * position_encoding_dimension
            elif position_encoding_kind == "gaussian":
                axis_scales = self._axis_scales(position_encoding_scale)
                self.pos_encoding = common.GaussianFourierFeatures(position_encoding_dimension, input_dim=3, scales=axis_scales, seed=position_encoding_seed)
                pos_size = position_encoding_dimension
            else:
                raise ValueError(f"Unknown position_encoding_kind: {position_encoding_kind!r}")
        else:
            pos_size = 0

        self.tau_encoding = common.FourierEncoding(tau_encoding_dimension)
        encoding_size = tau_encoding_dimension + 4 + 1 + pos_size

        equivariant_layers = [
            ("hidden0", self.get_layer(encoding_size, hidden_layer_size)),
            ("activation0", nn.ReLU())
        ]
        for ihidden in range(1, n_hidden_layers):
            equivariant_layers += [
                ("hidden{}".format(ihidden), self.get_layer(hidden_layer_size, hidden_layer_size)),
                ("activation{}".format(ihidden), nn.ReLU())
            ]

        self.predict_variances = predict_variances
        output_size = 4 * 2 if self.predict_variances else 4
        equivariant_layers.append(("prediction_output", self.get_layer(hidden_layer_size, output_size, initialize_zero=True)))

        self.prediction_tower = nn.Sequential(OrderedDict(equivariant_layers))

    def forward(self, input_set : torch.Tensor, tau : torch.Tensor):
        nhits = input_set.shape[-2]

        tau_encoded = self.tau_encoding(tau / self.n_timesteps)
        tau_encoded = tau_encoded.unsqueeze(-2).expand(*tau_encoded.shape[:-1], nhits, -1)

        nhits_feature = torch.log(input_set.new_full((*input_set.shape[:-1], 1), nhits / self.nhits_norm + 1))

        features = [tau_encoded, input_set[...,0:4], nhits_feature]
        if self.use_position_encoding:
            if self.position_encoding_kind == "gaussian":
                features.append(self.pos_encoding(input_set[...,1:4]))
            else:
                pos1_encoded = self.pos1_encoding(input_set[...,1])
                pos2_encoded = self.pos2_encoding(input_set[...,2])
                pos3_encoded = self.pos3_encoding(input_set[...,3])
                features += [pos1_encoded, pos2_encoded, pos3_encoded]

        encoded_set = torch.cat(features, axis=-1)
        out = self.prediction_tower(encoded_set)

        if self.predict_variances:
            delta = out[...,:4]
            variances = torch.nn.functional.softplus(out[..., 4:])
            return input_set + delta, variances

        return input_set + out
