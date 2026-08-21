"""Tests for Gaussian NLL loss."""

import pytest
import torch

from BIBgen.losses import *

def test_GaussianNLLLoss_full_shape_variance():
    """predict_variances=True path: variance shares mu/actual's full (N, 4) shape."""
    N = 100
    mu = torch.randn(N, 4, requires_grad=True)
    variance = (torch.rand(N, 4) + 0.1).requires_grad_()
    actual = torch.randn(N, 4)

    loss_fn = GaussianNLLLoss()
    loss = loss_fn(mu, variance, actual)

    assert loss.dim() == 0
    assert torch.isfinite(loss)

    loss.backward()
    assert torch.isfinite(mu.grad).all()
    assert torch.isfinite(variance.grad).all()

def test_GaussianNLLLoss_full_shape_variance_batched():
    B, N = 5, 24
    mu = torch.randn(B, N, 4, requires_grad=True)
    variance = (torch.rand(B, N, 4) + 0.1).requires_grad_()
    actual = torch.randn(B, N, 4)

    loss_fn = GaussianNLLLoss()
    loss = loss_fn(mu, variance, actual)

    assert loss.dim() == 0
    assert torch.isfinite(loss)

    loss.backward()
    assert torch.isfinite(mu.grad).all()
    assert torch.isfinite(variance.grad).all()

def test_GaussianNLLLoss_scalar_variance_matches_manual():
    """predict_variances=False, unbatched: one variance value shared by all hits/features."""
    N = 50
    mu = torch.randn(N, 4)
    actual = torch.randn(N, 4)
    variance = torch.tensor(0.3)

    loss = GaussianNLLLoss(eps=0.0)(mu, variance, actual)

    expected = 0.5 * (1.8378770664093453 + torch.log(variance) + (actual - mu) ** 2 / variance)
    assert loss.item() == pytest.approx(expected.mean().item(), abs=1e-5)

def test_GaussianNLLLoss_per_batch_variance_matches_manual():
    """predict_variances=False, batched: one variance value per event, shape (batch,)."""
    B, N = 4, 20
    mu = torch.randn(B, N, 4)
    actual = torch.randn(B, N, 4)
    variance = torch.rand(B) + 0.1

    loss = GaussianNLLLoss(eps=0.0)(mu, variance, actual)

    var_expanded = variance.view(B, 1, 1).expand(B, N, 4)
    expected_per_event = (0.5 * (1.8378770664093453 + torch.log(var_expanded)
                                  + (actual - mu) ** 2 / var_expanded)).mean(dim=(1, 2))
    assert loss.item() == pytest.approx(expected_per_event.sum().item(), abs=1e-5)
