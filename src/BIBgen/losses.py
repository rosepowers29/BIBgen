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
    
    def __init__(self, eps=1e-6, max_normalized_error=1e4):
        super().__init__()
        self.eps = eps
        # Hard ceiling on (actual-mu)^2 / variance, independent of how small variance or how bad
        # a single prediction happens to be. variance here is the noise schedule's fixed beta at
        # a given timestep, not something the model controls -- it has no way to adapt to how
        # well the model actually fits a given event. An ordinary-looking prediction error on an
        # unusually hard-to-fit event (confirmed directly in training: a max_squared_error of
        # just ~11 on an extreme-hit-count event, combined with one of the schedule's smaller-tau
        # variances, produced a normalized_error of ~8.9e4 and a non-finite gradient) can still
        # occasionally collide with a small-variance timestep and blow up, well after the point
        # in training where a quick check would otherwise suggest it's safe. This clamp bounds
        # that worst case directly, rather than relying on the schedule's own floor to keep every
        # possible squared_error/variance combination small -- 1e4 is a generous ceiling, ~1000x
        # the typical O(1-40) per-batch worst-case value seen in healthy training, so it leaves
        # room for real signal on hard examples while preventing a single pathological one from
        # producing a training-derailing update.
        self.max_normalized_error = max_normalized_error

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
        normalized_error = torch.clamp(squared_error / variance, max=self.max_normalized_error)
        
        # log(variance)
        log_variance = torch.log(variance)
        
        # NLL = 0.5 * (log(2pi) + log(variance) + normalized_error)
        loss = 0.5 * (1.8378770664093453 + log_variance + normalized_error)
        # Mean over the batch dimension (when present), not sum: a sum would make the loss --
        # and therefore gradient magnitude -- scale with whatever batch size a given step happens
        # to use, which varies enormously under BatchedDataLoader's hit_budget mode.
        return loss.mean(dim=(1, 2)).mean(dim=0) if loss.dim() > 2 else loss.mean()

    def diagnostics(self, mu, variance, actual):
        """
        Returns the batch's worst-case (max, not mean) squared_error and normalized_error,
        without affecting the loss computation itself. The mean loss returned by forward() can
        hide a single outsized per-hit contribution -- useful for catching which timestep/event
        is actually responsible for a large or non-finite gradient.
        """
        with torch.no_grad():
            variance = variance + self.eps
            squared_error = (actual - mu) ** 2
            if variance.shape != squared_error.shape:
                variance = variance.unsqueeze(-1).unsqueeze(-1).expand(*variance.shape, *squared_error.shape[-2:])
            normalized_error = squared_error / variance
            return {
                "max_squared_error": squared_error.max().item(),
                "max_normalized_error": normalized_error.max().item(),
            }


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
