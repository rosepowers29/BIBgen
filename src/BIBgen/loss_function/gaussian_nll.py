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
        normalized_error = squared_error / variance
        
        # log(variance)
        log_variance = torch.log(variance)
        
        # NLL = 0.5 * (log(2pi) + log(variance) + normalized_error)
        loss = 0.5 * (1.8378770664093453 + log_variance + normalized_error)
    

        return loss.mean()
