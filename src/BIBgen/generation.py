import torch
from torch import nn

def generate_sphered(
    model : nn.Module,
    n_members : int,
    device : torch.device,
    schedule : torch.Tensor | None = None,
    demo : bool = False,
    verbosity : int = 0
):
    """
    Run the denoising procedure on white noise.
    Does not reverse the preprocessing on the output in any way.
    Requires a fixed number of hits

    Parameters
    ----------
    model : nn.Module
        Denoising model
    n_members : int
        Number of hits to appear in this event
    device : torch.Device
        device on which to perform, usually torch.device("cuda") or torch.device("cpu")
    schedule : torch.Tensor | None
        If model does not predict variances, then the noise schedule msut be provided in lieu of it
    demo : bool
        While denoising, whether to sample from N(mu, std) or just choose mu. Note setting the latter often risks collapse of the distribution.
    verbosity : bool
        Print diagnostics for white noise and step wise results

    Returns
    -------
    event : torch.Tensor
        The generated event with shape (n_members, 4)

    Raises
    ------
    ValueError
        If model does not predict variances and schedule is not provided
    RuntimeError
        If variances predicted by model become non-finite
    """
    model = model.to(device)
    current_event = torch.normal(mean=0, std=1, size=(n_members, 4), device=device)
    if verbosity >= 1:
        print("White noise:", current_event[:10])

    with torch.inference_mode():
        for tau in torch.arange(model.n_timesteps-1, -1, -1, device=device):
            result = model(current_event, tau)

            if schedule is None:
                if len(result) != 2:
                    raise ValueError("If schedule is not provided, result should be mu and var")
                mu, var = result
                if not torch.isfinite(var).all():
                    print("Non-finite var at tau", tau, flush=True)
                    raise RuntimeError("var has NaN/Inf")
                if (var <= 0).any():
                    print("Non-positive var at tau", tau, "min", var.min().item(), flush=True)
            else:
                mu = result
                var = schedule[tau]

            current_event = torch.normal(mean=mu, std=torch.sqrt(var)) if not demo else mu

            if verbosity >= 2:
                print("tau", tau,
                "x max", current_event.abs().max().item(),
                "x mean", current_event.abs().mean().item(),
                flush=True)

    return current_event