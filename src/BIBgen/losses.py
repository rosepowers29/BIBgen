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
        - mu: mean of each hit (N, 4)
        - variance: uncertainty in each component (N, 4)
    
    Loss is computed as -log P(actual | mu, variance) where P is Gaussian.
    """
    
    def __init__(self, eps=1e-6):
        super().__init__()
        self.eps = eps
    
    def forward(self, mu, variance, actual):
        """
        Args:
            mu: predicted means (N, 4)
            variance: predicted variances (N, 4) 
            actual: ground truth (N, 4)
        
        Returns:
            mean loss over the batch
        """
        # Add small constant for stability
        variance = variance + self.eps
        
        # (actual - mu)^2 / variance
        squared_error = (actual - mu) ** 2
        if variance.shape != squared_error.shape:
            variance = variance.unsqueeze(-1).unsqueeze(-1).expand(*variance.shape, *squared_error.shape[-2:])
        normalized_error = (squared_error / variance)
        
        # log(variance)
        log_variance = torch.log(variance)
        
        # NLL = 0.5 * (log(2pi) + log(variance) + normalized_error)
        loss = 0.5 * (1.8378770664093453 + log_variance + normalized_error)
        return loss.mean(dim=(1, 2)).sum(dim=0) if loss.dim() > 2 else loss.mean()


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

    def __init__(self, eps=1e-6, variance_loss_weight=1.0):
        super().__init__()
        self.nll = GaussianNLLLoss(eps=eps)
        self.variance_loss_weight = variance_loss_weight

    def forward(self, mu, variance, actual):
        mean_loss = self.nll(mu, variance.detach(), actual)
        variance_loss = self.nll(mu.detach(), variance, actual)
        return mean_loss + self.variance_loss_weight * variance_loss
