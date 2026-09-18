"""
Gaussian negative log-likelihood loss for BIBgen.

Computes -log P(actual | mu, sigma) for training the diffusion model
with uncertainty estimates.
"""

import torch
import torch.nn as nn


def _gaussian_nll_elementwise(mu, variance, actual, eps):
    """
    Per-element Gaussian NLL, unreduced. Shared by GaussianNLLLoss (which reduces it directly)
    and NELBOLoss (which must mix it elementwise with a KL term via torch.where before reducing).
    """
    variance = variance + eps

    squared_error = (actual - mu) ** 2
    if variance.shape != squared_error.shape:
        variance = variance.unsqueeze(-1).unsqueeze(-1).expand(*variance.shape, *squared_error.shape[-2:])
    normalized_error = squared_error / variance

    log_variance = torch.log(variance)
    return 0.5 * (1.8378770664093453 + log_variance + normalized_error)


def _reduce_per_element_loss(loss):
    return loss.mean(dim=(1, 2)).sum(dim=0) if loss.dim() > 2 else loss.mean()


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
        return _reduce_per_element_loss(_gaussian_nll_elementwise(mu, variance, actual, self.eps))


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


class NELBOLoss(nn.Module):
    r"""
    Negative ELBO loss (Ho et al. 2020, "Denoising Diffusion Probabilistic Models"),
    reindexed to this codebase's 0-indexed $\beta_\tau$ convention ($\beta_\tau$ governs the
    transition $x_\tau \to x_{\tau+1}$; $\tau=0$ is clean data).

    Unlike GaussianNLLLoss/DecoupledGaussianNLLLoss -- which score the model's prediction
    directly against the one observed $x_\tau$ from a specific forward-diffusion sample --
    this compares the model's predicted reverse distribution $p_\theta(x_\tau|x_{\tau+1})$
    against the *true* forward-process posterior $q(x_\tau|x_{\tau+1},x_0)$ via the closed-form
    Gaussian KL divergence, for $\tau \geq 1$:

        $\bar\alpha_\tau = \prod_{s=0}^{\tau-1}(1-\beta_s)$
        $q(x_\tau|x_{\tau+1},x_0) = \mathcal{N}(\tilde\mu_\tau, \tilde\beta_\tau)$
        $\tilde\mu_\tau = \frac{\sqrt{\bar\alpha_\tau}\,\beta_\tau}{1-\bar\alpha_{\tau+1}} x_0
                        + \frac{\sqrt{1-\beta_\tau}\,(1-\bar\alpha_\tau)}{1-\bar\alpha_{\tau+1}} x_{\tau+1}$
        $\tilde\beta_\tau = \frac{1-\bar\alpha_\tau}{1-\bar\alpha_{\tau+1}} \beta_\tau$

    (Derived directly via precision-weighted combination of $p(x_\tau|x_0)$ and
    $p(x_{\tau+1}|x_\tau)$, not just reindexed by analogy -- confirmed to match Ho et al. eq. 6-7
    under the index shift their $t$ = this codebase's $\tau+1$.)

    At $\tau=0$, $\tilde\beta_0=0$ exactly (x_0 is already known, so the posterior degenerates to
    a point mass) -- plugging that into the general KL formula blows up via $\log\tilde\beta_0$,
    so DDPM instead uses a direct reconstruction term $-\log p_\theta(x_0|x_1)$ there, which is
    exactly the existing Gaussian NLL evaluated against the actual $x_0$.

    For predict_variances=False models, $p_\theta$'s variance is fixed to the schedule's own
    $\beta_\tau$ (matching GaussianNLLLoss's existing convention) rather than a learned value.
    """

    def __init__(self, betas, eps=1e-6, variance_loss_weight=None):
        """
        Parameters
        ----------
        betas : array-like
            Noise schedule, shape (n_timesteps,) -- same array used for forward diffusion.
        eps : float
            Numerical floor added to variances before taking logs/reciprocals.
        variance_loss_weight : float | None
            If not None, applies the same mean/variance gradient decoupling as
            DecoupledGaussianNLLLoss (only meaningful when the model predicts variances):
            the loss is computed twice, once with variance detached (mean-training term) and
            once with mu detached (variance-training term), combined as
            `mean_loss + variance_loss_weight * variance_loss`. None (default) trains mu and
            variance jointly, undecoupled.
        """
        super().__init__()
        betas = torch.as_tensor(betas, dtype=torch.float32)
        self.register_buffer("betas", betas)
        self.register_buffer("alpha_bar", torch.cat([torch.ones(1), torch.cumprod(1 - betas, dim=0)]))
        self.eps = eps
        self.variance_loss_weight = variance_loss_weight

    @staticmethod
    def _reshape_like(value, ref):
        """Pads trailing singleton dims onto a per-tau scalar/vector so it broadcasts against (..., n_hits, 4)-shaped tensors."""
        while value.dim() < ref.dim():
            value = value.unsqueeze(-1)
        return value

    def posterior(self, x_0, x_tau_plus1, tau):
        """
        Closed-form true posterior q(x_tau | x_{tau+1}, x_0).

        Returns
        -------
        mean : torch.Tensor
            Same shape as x_tau_plus1.
        variance : torch.Tensor
            Broadcastable against x_tau_plus1's shape; scalar-per-(batch-)item, not per-feature.
        """
        beta_tau = self._reshape_like(self.betas[tau], x_tau_plus1)
        alpha_bar_tau = self._reshape_like(self.alpha_bar[tau], x_tau_plus1)
        alpha_bar_tau_plus1 = self._reshape_like(self.alpha_bar[tau + 1], x_tau_plus1)

        mean = (
            (torch.sqrt(alpha_bar_tau) * beta_tau / (1 - alpha_bar_tau_plus1)) * x_0
            + (torch.sqrt(1 - beta_tau) * (1 - alpha_bar_tau) / (1 - alpha_bar_tau_plus1)) * x_tau_plus1
        )
        variance = (1 - alpha_bar_tau) / (1 - alpha_bar_tau_plus1) * beta_tau
        return mean, variance

    def _elementwise(self, mu_theta, var_theta_raw, x_tau_plus1, tau, x_0):
        posterior_mean, posterior_var = self.posterior(x_0, x_tau_plus1, tau)

        var_theta = self._reshape_like(var_theta_raw, mu_theta) + self.eps
        posterior_var_safe = posterior_var + self.eps

        kl = 0.5 * (
            torch.log(var_theta) - torch.log(posterior_var_safe)
            + (posterior_var + (posterior_mean - mu_theta) ** 2) / var_theta
            - 1.0
        )
        reconstruction = _gaussian_nll_elementwise(mu_theta, var_theta_raw, x_0, self.eps)

        is_zero = self._reshape_like(tau == 0, mu_theta)
        return torch.where(is_zero, reconstruction, kl)

    def forward(self, pred, x_tau_plus1, tau, x_0):
        """
        Parameters
        ----------
        pred : torch.Tensor | tuple[torch.Tensor, torch.Tensor]
            Model output: mu (predict_variances=False) or (mu, variance) (predict_variances=True).
        x_tau_plus1 : torch.Tensor
            The noisier input the model conditioned on to produce pred (i.e. x_{tau+1}).
        tau : torch.Tensor
            Diffusion timestep index (0-d scalar shared across the batch, or one per batch item).
        x_0 : torch.Tensor
            The clean sample the forward-diffusion trajectory started from.
        """
        if isinstance(pred, tuple):
            mu_theta, var_theta_raw = pred
        else:
            mu_theta, var_theta_raw = pred, self.betas[tau]

        if self.variance_loss_weight is None or not isinstance(pred, tuple):
            return _reduce_per_element_loss(self._elementwise(mu_theta, var_theta_raw, x_tau_plus1, tau, x_0))

        mean_loss = _reduce_per_element_loss(
            self._elementwise(mu_theta, var_theta_raw.detach(), x_tau_plus1, tau, x_0))
        variance_loss = _reduce_per_element_loss(
            self._elementwise(mu_theta.detach(), var_theta_raw, x_tau_plus1, tau, x_0))
        return mean_loss + self.variance_loss_weight * variance_loss
