"""Test for Gaussian NLL loss."""

import torch
from BIBgen.loss_function.gaussian_nll import GaussianNLLLoss


if __name__ == "__main__":
    # Quick test
    N = 100
    mu = torch.randn(N, 4, requires_grad=True)
    variance = torch.rand(N, 4, requires_grad=True) + 0.1
    actual = torch.randn(N, 4)
    
    loss_fn = GaussianNLLLoss()
    loss = loss_fn(mu, variance, actual)
    
    print(f"Loss: {loss.item():.4f}")
    
    loss.backward()
    print(f"Grad norm (mu): {mu.grad.norm().item():.4f}")
    print(f"Grad norm (var): {variance.grad.norm().item():.4f}")
    print("Test passed!")
