"""
Gaussian negative log-likelihood loss for BIBgen.

Computes -log P(actual | mu, sigma) for training the diffusion model
with uncertainty estimates.
"""

import torch
import torch.nn as nn


class GaussianNLLLoss(nn.Module):
    """
    NLL loss assuming diagonal covariance.
    
    The model predicts:
        - mu: mean of each hit (N, N_SCALAR_FEATURES)
        - variance: uncertainty in each component (N, N_SCALAR_FEATURES)
    
    Loss is computed as -log P(actual | mu, variance) where P is Gaussian.
    """
    
    def __init__(self, eps=1e-9, circular_std=None, circular_index=1):
        """
        Args:
            circular_std: standard deviation used to normalize the circular feature (e.g.
                the saved Sphering std for phi), or None if the data has no wrapped feature
                (e.g. cartesian-coordinate training data, where column `circular_index` is
                an unbounded x rather than an angle). Differences along `circular_index` are
                measured in un-normalized (raw) units via this std, wrapped to (-pi, pi], then
                re-normalized -- so e.g. a raw phi of 2*pi - 0.1 and 0.1 are treated as 0.2
                apart rather than 2*pi - 0.2.
            circular_index: feature index holding the wrapped angle, only used when
                circular_std is not None. Default 1 matches the [energy, phi, s, z, time]
                cylindrical feature layout (see features.py).
        """
        super().__init__()
        self.eps = eps
        self.circular_std = circular_std
        self.circular_index = circular_index

    def forward(self, mu, variance, actual):
        """
        Args:
            mu: predicted means (N, N_SCALAR_FEATURES)
            variance: predicted variances (N, N_SCALAR_FEATURES)
            actual: ground truth (N, N_SCALAR_FEATURES)

        Returns:
            mean loss over the batch
        """
        # Add small constant for stability
        variance = variance + self.eps

        diff = actual - mu
        if self.circular_std is not None:
            i = self.circular_index
            raw_diff = diff[..., i] * self.circular_std
            wrapped_raw_diff = torch.atan2(torch.sin(raw_diff), torch.cos(raw_diff))
            diff = torch.cat(
                [diff[..., :i], (wrapped_raw_diff / self.circular_std).unsqueeze(-1), diff[..., i + 1:]],
                dim=-1,
            )

        # (actual - mu)^2 / variance
        squared_error = diff ** 2
        if variance.shape != squared_error.shape:
            variance = variance.unsqueeze(-1).unsqueeze(-1).expand(*variance.shape, *squared_error.shape[-2:])
        normalized_error = (squared_error / variance)
        
        # log(variance)
        log_variance = torch.log(variance)
        
        # NLL = 0.5 * (log(2pi) + log(variance) + normalized_error)
        loss = 0.5 * (1.8378770664093453 + log_variance + normalized_error)
        # Mean over the batch dimension (when present), not sum: a sum would make the loss --
        # and therefore gradient magnitude -- scale with whatever batch size a given step happens
        # to use, which varies enormously under BatchedDataLoader's hit_budget mode.
        return loss.mean(dim=(1, 2)).mean(dim=0) if loss.dim() > 2 else loss.mean()


class DecoupledGaussianNLLLoss(nn.Module):
    """
    Gaussian NLL with mean/variance gradient decoupling (Nichol & Dhariwal, "Improved DDPM").

    Splits the loss into a mean-training term (variance detached) and a variance-training
    term (mean detached), so fitting the variance can't perturb the mean-prediction path
    through a shared network trunk, and vice versa -- preventing the variance head from
    collapsing onto a trivial, input-independent solution due to noisy/conflicting shared-trunk
    gradients. Note this does not fully isolate the two: when mu and variance share every
    hidden layer (as in EquivariantDenoiser), the shared trunk still receives gradient from
    variance_loss via variance's own (non-detached) output path -- the detach specifically
    prevents variance_loss's gradient from also flowing into the trunk via the mean's path,
    and vice versa, avoiding the double/conflicting-signal problem.
    """

    def __init__(self, eps=1e-6, variance_loss_weight=1.0, circular_std=None, circular_index=1):
        super().__init__()
        self.nll = GaussianNLLLoss(eps=eps, circular_std=circular_std, circular_index=circular_index)
        self.variance_loss_weight = variance_loss_weight

    def forward(self, mu, variance, actual):
        mean_loss = self.nll(mu, variance.detach(), actual)
        variance_loss = self.nll(mu.detach(), variance, actual)
        return mean_loss + self.variance_loss_weight * variance_loss
