"""Tests for Gaussian NLL loss."""

import pytest
import numpy as np
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

def test_NELBOLoss_posterior_matches_manual():
    """q(x_tau | x_{tau+1}, x_0) closed form, independently recomputed (not via NELBOLoss internals)."""
    betas = torch.tensor([0.1, 0.2, 0.3])
    N = 20
    x_0 = torch.randn(N, 4)
    x_tau_plus1 = torch.randn(N, 4)
    tau = torch.tensor(1)

    nelbo = NELBOLoss(betas)
    mean, variance = nelbo.posterior(x_0, x_tau_plus1, tau)

    alpha_bar_1, alpha_bar_2 = 0.9, 0.9 * 0.8
    beta_1 = 0.2
    expected_mean = (
        (alpha_bar_1 ** 0.5 * beta_1 / (1 - alpha_bar_2)) * x_0
        + ((1 - beta_1) ** 0.5 * (1 - alpha_bar_1) / (1 - alpha_bar_2)) * x_tau_plus1
    )
    expected_variance = (1 - alpha_bar_1) / (1 - alpha_bar_2) * beta_1

    assert mean.numpy() == pytest.approx(expected_mean.numpy(), abs=1e-5)
    assert variance.item() == pytest.approx(expected_variance, abs=1e-6)

def test_NELBOLoss_posterior_degenerates_at_tau_zero():
    """q(x_0 | x_1, x_0) is a point mass at x_0 -- mean=x_0 exactly, variance=0 exactly."""
    betas = torch.tensor([0.1, 0.2, 0.3])
    x_0 = torch.randn(20, 4)
    x_1 = torch.randn(20, 4)

    nelbo = NELBOLoss(betas)
    mean, variance = nelbo.posterior(x_0, x_1, torch.tensor(0))

    assert mean.numpy() == pytest.approx(x_0.numpy(), abs=1e-5)
    assert variance.item() == pytest.approx(0.0, abs=1e-6)

def test_NELBOLoss_tau_zero_matches_reconstruction_term():
    """At tau=0 the KL branch is replaced by the plain Gaussian NLL reconstruction term against x_0."""
    betas = torch.tensor([0.1, 0.2, 0.3])
    N = 20
    mu_theta = torch.randn(N, 4)
    x_0 = torch.randn(N, 4)
    x_1 = torch.randn(N, 4)  # arbitrary; unused by the tau=0 branch

    nelbo = NELBOLoss(betas, eps=0.0)
    loss = nelbo(mu_theta, x_1, torch.tensor(0), x_0)

    expected = GaussianNLLLoss(eps=0.0)(mu_theta, betas[0], x_0)
    assert loss.item() == pytest.approx(expected.item(), abs=1e-5)

def test_NELBOLoss_predict_variances_tuple_input():
    """predict_variances=True path: pred is a (mu, variance) tuple with full (N, 4) variance shape."""
    betas = torch.tensor([0.1, 0.2, 0.3, 0.1, 0.2])
    N = 20
    mu_theta = torch.randn(N, 4, requires_grad=True)
    var_theta = (torch.rand(N, 4) + 0.1).requires_grad_()
    x_0 = torch.randn(N, 4)
    x_tau_plus1 = torch.randn(N, 4)

    nelbo = NELBOLoss(betas)
    loss = nelbo((mu_theta, var_theta), x_tau_plus1, torch.tensor(2), x_0)

    assert loss.dim() == 0
    assert torch.isfinite(loss)
    loss.backward()
    assert torch.isfinite(mu_theta.grad).all()
    assert torch.isfinite(var_theta.grad).all()

def test_NELBOLoss_batched_matches_per_sample():
    """A batch of distinct tau values must give the same per-element loss as looping one tau at a time."""
    betas = torch.tensor([0.1, 0.2, 0.3, 0.15, 0.25])
    N = 10
    B = 5
    mu_theta = torch.randn(B, N, 4)
    x_0 = torch.randn(N, 4)
    x_tau_plus1 = torch.randn(B, N, 4)
    tau = torch.tensor([0, 1, 2, 3, 4])

    nelbo = NELBOLoss(betas)
    batched = nelbo._elementwise(mu_theta, betas[tau], x_tau_plus1, tau, x_0)

    for i in range(B):
        per_sample = nelbo._elementwise(mu_theta[i], betas[tau[i]], x_tau_plus1[i], tau[i], x_0)
        assert batched[i].numpy() == pytest.approx(per_sample.numpy(), abs=1e-5)

def test_NELBOLoss_decoupling_mu_grad_matches_mean_term_alone():
    betas = torch.tensor([0.1, 0.2, 0.3])
    N = 20
    mu_theta = torch.randn(N, 4, requires_grad=True)
    var_theta = (torch.rand(N, 4) + 0.1).requires_grad_()
    x_0 = torch.randn(N, 4)
    x_tau_plus1 = torch.randn(N, 4)
    tau = torch.tensor(1)

    NELBOLoss(betas, variance_loss_weight=1.0)((mu_theta, var_theta), x_tau_plus1, tau, x_0).backward()
    decoupled_mu_grad = mu_theta.grad.clone()

    mu_theta.grad = None
    NELBOLoss(betas)((mu_theta, var_theta.detach()), x_tau_plus1, tau, x_0).backward()

    assert decoupled_mu_grad == pytest.approx(mu_theta.grad, abs=1e-6)

def test_NELBOLoss_decoupling_zero_weight_zeroes_variance_grad():
    betas = torch.tensor([0.1, 0.2, 0.3])
    N = 20
    mu_theta = torch.randn(N, 4, requires_grad=True)
    var_theta = (torch.rand(N, 4) + 0.1).requires_grad_()
    x_0 = torch.randn(N, 4)
    x_tau_plus1 = torch.randn(N, 4)
    tau = torch.tensor(1)

    loss = NELBOLoss(betas, variance_loss_weight=0.0)((mu_theta, var_theta), x_tau_plus1, tau, x_0)
    loss.backward()

    assert torch.isfinite(mu_theta.grad).all()
    assert (var_theta.grad == 0).all()

def test_NELBOLoss_through_equivariant_denoiser():
    """Integration check: NELBOLoss produces finite gradients through the real shared-trunk architecture, on an actual diffused trajectory."""
    from BIBgen.models import EquivariantDenoiser
    from BIBgen.preprocessing import diffuse

    betas = np.linspace(0.05, 0.2, 10)
    trajectory = diffuse(np.random.rand(24, 4), betas)  # shape (11, 24, 4)
    x_0 = torch.from_numpy(trajectory[0]).to(dtype=torch.float32)
    tau = torch.tensor(4)
    x_tau_plus1 = torch.from_numpy(trajectory[tau.item() + 1]).to(dtype=torch.float32)

    model = EquivariantDenoiser(
        n_timesteps=10,
        tau_encoding_dimension=8,
        position_encoding_dimension=8,
        hidden_layer_size=32,
        n_hidden_layers=1,
        predict_variances=True,
    )

    pred = model(x_tau_plus1, tau=tau)
    nelbo = NELBOLoss(betas, variance_loss_weight=1.0)
    loss = nelbo(pred, x_tau_plus1, tau, x_0)
    loss.backward()

    for name, p in model.named_parameters():
        assert p.grad is not None and torch.isfinite(p.grad).all(), name
