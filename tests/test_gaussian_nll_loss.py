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

def test_DecoupledGaussianNLLLoss_mu_grad_matches_mean_term_alone():
    """variance_loss's contribution to mu's gradient must be exactly zero."""
    N = 50
    mu = torch.randn(N, 4, requires_grad=True)
    variance = (torch.rand(N, 4) + 0.1).requires_grad_()
    actual = torch.randn(N, 4)

    DecoupledGaussianNLLLoss()(mu, variance, actual).backward()
    decoupled_mu_grad = mu.grad.clone()

    mu.grad = None
    variance_detached = variance.detach()
    GaussianNLLLoss()(mu, variance_detached, actual).backward()

    assert decoupled_mu_grad == pytest.approx(mu.grad, abs=1e-6)

def test_DecoupledGaussianNLLLoss_variance_grad_matches_variance_term_alone():
    """mean_loss's contribution to variance's gradient must be exactly zero."""
    N = 50
    mu = torch.randn(N, 4, requires_grad=True)
    variance = (torch.rand(N, 4) + 0.1).requires_grad_()
    actual = torch.randn(N, 4)

    DecoupledGaussianNLLLoss()(mu, variance, actual).backward()
    decoupled_variance_grad = variance.grad.clone()

    variance.grad = None
    mu_detached = mu.detach()
    GaussianNLLLoss()(mu_detached, variance, actual).backward()

    assert decoupled_variance_grad == pytest.approx(variance.grad, abs=1e-6)

def test_DecoupledGaussianNLLLoss_zero_weight_zeroes_variance_grad():
    N = 50
    mu = torch.randn(N, 4, requires_grad=True)
    variance = (torch.rand(N, 4) + 0.1).requires_grad_()
    actual = torch.randn(N, 4)

    loss = DecoupledGaussianNLLLoss(variance_loss_weight=0.0)(mu, variance, actual)
    loss.backward()

    assert torch.isfinite(mu.grad).all()
    assert (variance.grad == 0).all()

def test_DecoupledGaussianNLLLoss_through_equivariant_denoiser():
    """Integration check: decoupled loss produces finite gradients through the real shared-trunk architecture."""
    from BIBgen.models import EquivariantDenoiser

    model = EquivariantDenoiser(
        n_timesteps=25,
        tau_encoding_dimension=8,
        position_encoding_dimension=8,
        hidden_layer_size=32,
        n_hidden_layers=1,
        predict_variances=True,
    )

    tau = torch.tensor(12)
    input_set = torch.rand((24, 4))
    actual = torch.rand((24, 4))

    mu, variance = model(input_set, tau=tau)
    loss = DecoupledGaussianNLLLoss()(mu, variance, actual)
    loss.backward()

    for name, p in model.named_parameters():
        assert p.grad is not None and torch.isfinite(p.grad).all(), name
