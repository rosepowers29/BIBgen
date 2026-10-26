from typing import Callable

import torch
import numpy as np

from BIBgen.training.dataloaders import BaseDataLoader

def log_per_layer_grads(model):
    for name, p in model.named_parameters():
        if p.grad is not None:
            print(f"  {name:45s} grad={p.grad.norm().item():.3e}  param={p.norm().item():.3e}")

def train(
    dataloader : BaseDataLoader,
    model : torch.nn.Module,
    loss_fn : Callable,
    optimizer : torch.optim.Optimizer,
    device : torch.device,
    scaler : torch.amp.GradScaler | None = None,
    accumulation_steps : int = 1,
    max_steps_diagnostics : int = 0
):
    """
    Trains the model for one epoch.

    Parameters
    ----------
    dataloader : DataLoader
        dataloader for training data
    model : torch.nn.Module
        Denoising model
    loss_fn : Callable
        loss function to be called directly on model output, i.e. loss_fn(pred, y, tau) instead of loss_fn(mu, std, y)
    optimizer : torch.optim.Optimizer
        Optimizer for gradient descent
    device : torch.device
        device on which to perform, usually torch.device("cuda") or torch.device("cpu")
    accumulation_steps : int
        Split each dataloader batch into this many equal micro-batches, backpropagating
        each separately and only stepping the optimizer once per full batch. Reduces peak
        activation memory by this factor while keeping the effective batch size (and the
        resulting gradient) unchanged. The dataloader's batch size must be divisible by it.
    max_steps_diagnostics : int
        Maximum number of steps to print diagnostics

    Returns
    -------
    loss : float
        Training loss on the final micro-batch of the final iteration
    """
    # assert scaler is not None or device.type != "cuda"
    model.train()
    grad_norm = None
    for istep in range(dataloader.nsteps):
        X, y, tau = next(dataloader)
        X, y, tau = X.to(device), y.to(device), tau.to(device)

        if accumulation_steps > 1 and X.shape[0] % accumulation_steps != 0:
            raise ValueError("Batch size must be divisible by accumulation_steps")
        micro_batch_size = X.shape[0] // accumulation_steps if accumulation_steps > 1 else None

        optimizer.zero_grad(set_to_none=True)

        try:
            for micro_step in range(accumulation_steps):
                if accumulation_steps == 1:
                    # Unsliced: also correct for UnbatchedDataLoader, whose X/y/tau have no
                    # leading batch dimension to slice (tau is a 0-dim scalar tensor there).
                    X_sub, y_sub, tau_sub = X, y, tau
                else:
                    sl = slice(micro_step * micro_batch_size, (micro_step + 1) * micro_batch_size)
                    X_sub, y_sub, tau_sub = X[sl], y[sl], tau[sl]

                # Compute prediction error
                if scaler is not None:
                    with torch.autocast(device_type='cuda', dtype=torch.float16):
                        pred = model(X_sub, tau_sub)
                        loss = loss_fn(pred, y_sub, tau_sub)
                else:
                    pred = model(X_sub, tau_sub)
                    loss = loss_fn(pred, y_sub, tau_sub)

                # loss_fn reduces with a sum over the batch dimension (see GaussianNLLLoss),
                # not a mean -- so summing each micro-batch's backward (via repeated .backward()
                # calls accumulating into .grad) exactly reproduces the full batch's gradient,
                # with no rescaling needed.
                if scaler is not None:
                    scaler.scale(loss).backward()
                else:
                    loss.backward()

            if scaler is not None:
                scaler.step(optimizer)
                scaler.update()
            else:
                if istep < max_steps_diagnostics:
                    log_per_layer_grads(model)
                optimizer.step()
        except torch.cuda.OutOfMemoryError:
            print("Failed step {}, cuda out of memory".format(istep))
            print("X.size() : {}, y.size() : {}, accumulation_steps : {}".format(X.size(), y.size(), accumulation_steps))
            if isinstance(pred, tuple):
                print("pred[0].size() (mu) : {}, pred[1].size() (var) : {}".format(pred[0].size(), pred[1].size()))
            else:
                print("pred.size() : {}".format(pred.size()))
            raise RuntimeError("Intentional exit")

    return loss.item()

def evaluate(
    dataloader : BaseDataLoader,
    model : torch.nn.Module,
    loss_fn : Callable,
    device : torch.device,
    use_amp : bool = False,
    accumulation_steps : int = 1,
):
    """
    Parameters
    ----------
    dataloader : DataLoader
        dataloader for training data
    model : torch.nn.Module
        Denoising model
    loss_fn : Callable
        loss function to be called directly on model output, i.e. loss_fn(pred, y, tau) instead of loss_fn(mu, std, y)
    device : torch.device
        device on which to perform, usually torch.device("cuda") or torch.device("cpu")
    use_amp : bool
        Run the forward pass under torch.autocast(dtype=torch.float16) to reduce GPU memory use
    accumulation_steps : int
        Split each dataloader batch into this many equal micro-batches, evaluated one at a
        time, to reduce peak activation memory. Purely a memory knob -- does not change the
        returned loss (no gradients are computed here, so there is nothing to accumulate).
        The dataloader's batch size must be divisible by it.

    Returns
    -------
    test_loss : float
        Average loss over all validation events
    """

    model.eval()
    test_loss = 0
    with torch.no_grad():
        for istep in range(dataloader.nsteps):
            X, y, tau = next(dataloader)
            X, y, tau = X.to(device), y.to(device), tau.to(device)

            if accumulation_steps > 1 and X.shape[0] % accumulation_steps != 0:
                raise ValueError("Batch size must be divisible by accumulation_steps")
            micro_batch_size = X.shape[0] // accumulation_steps if accumulation_steps > 1 else None

            step_loss = 0.0
            for micro_step in range(accumulation_steps):
                if accumulation_steps == 1:
                    X_sub, y_sub, tau_sub = X, y, tau
                else:
                    sl = slice(micro_step * micro_batch_size, (micro_step + 1) * micro_batch_size)
                    X_sub, y_sub, tau_sub = X[sl], y[sl], tau[sl]

                if use_amp:
                    with torch.autocast(device_type='cuda', dtype=torch.float16):
                        pred = model(X_sub, tau_sub)
                        micro_loss = loss_fn(pred, y_sub, tau_sub)
                else:
                    pred = model(X_sub, tau_sub)
                    micro_loss = loss_fn(pred, y_sub, tau_sub)
                step_loss += micro_loss.item()

            # loss_fn sums over the batch dimension (see GaussianNLLLoss), so summing each
            # micro-batch's loss exactly reproduces the full batch's loss -- no rescaling needed.
            test_loss += step_loss

            if not np.isfinite(test_loss):
                print("test_loss:", test_loss)
                print("X:", X)
                print("y:", y)
                print("pred:", pred)
                raise RuntimeError("Nonfinite test loss")

    return test_loss / dataloader.nsteps
