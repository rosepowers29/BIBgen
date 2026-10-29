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
    max_steps_diagnostics : int = 0,
    log_every : int = 0,
    max_grad_norm : float | None = None,
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
    log_every : int
        Print a one-line progress message every this many steps (batch size, hit count, running
        loss). 0 disables it. Per-event hit counts in this dataset range over two orders of
        magnitude, so a step can silently take much longer than its neighbors -- this is mainly
        useful for telling "still working on a big event" apart from "actually stuck" when
        watching a run live (e.g. on an interactive node), without flooding a long batch job's log.
    max_grad_norm : float | None
        If given, clip each step's (post-accumulation) global gradient norm to this value before
        the optimizer step. Guards against a single pathological batch (e.g. a numerically rough
        outlier event, more so under --amp's narrower fp16 range) applying an outsized update
        that derails training for many subsequent epochs. None (default) disables clipping.

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

        if log_every and istep % log_every == 0:
            print(f"[train] step {istep + 1}/{dataloader.nsteps}: batch_size={X.shape[0]}, nhits={X.shape[-2]}", flush=True)

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

                # loss_fn reduces with a mean over the batch, so each micro-batch's loss is
                # already on a per-sample scale regardless of its own size. Dividing by
                # accumulation_steps before backward (then summing via repeated .backward() calls
                # accumulating into .grad) reconstructs the gradient of the true full-batch mean,
                # since the micro-batches are equal-sized: mean of means over equal-size groups
                # is the overall mean exactly.
                if scaler is not None:
                    scaler.scale(loss / accumulation_steps).backward()
                else:
                    (loss / accumulation_steps).backward()

            want_grad_norm = max_grad_norm is not None or istep < max_steps_diagnostics
            if want_grad_norm:
                if scaler is not None:
                    # Gradients are still scaled by scaler's factor at this point; measuring or
                    # clipping against that would compare against values inflated by the
                    # (dynamic) scale factor rather than the true gradient magnitude.
                    scaler.unscale_(optimizer)
                # max_norm=inf is a standard "measure, don't modify" trick when only diagnostics
                # were requested: clip_coef is always >= 1, so no scaling is ever applied.
                total_norm = torch.nn.utils.clip_grad_norm_(
                    model.parameters(), max_grad_norm if max_grad_norm is not None else float("inf")
                )

            if istep < max_steps_diagnostics:
                # Previously nested under the no-AMP branch below, so it never printed under
                # --amp at all (scaler is never None there) -- now runs regardless of AMP, and
                # p.grad reflects the unscaled gradient either way thanks to the unscale_ above.
                log_per_layer_grads(model)
                if want_grad_norm:
                    norm_value = total_norm.item()
                    if not np.isfinite(norm_value):
                        # Expected during AMP's early scale-discovery: GradScaler already
                        # detected this overflow during unscale_ above (independently of this
                        # diagnostic) and will skip the optimizer step below and back off its
                        # scale factor -- not a sign training itself is unstable, unless it keeps
                        # happening well past the first several steps.
                        note = " (AMP scale backoff: GradScaler already caught this -- optimizer step skipped, not applied)"
                    else:
                        note = f" (clipped to {max_grad_norm})" if max_grad_norm is not None and norm_value > max_grad_norm else ""
                    print(f"  global grad norm: {norm_value:.3e}{note}")

            if scaler is not None:
                scaler.step(optimizer)
                scaler.update()
            else:
                optimizer.step()
        except torch.cuda.OutOfMemoryError:
            print("Failed step {}, cuda out of memory".format(istep))
            print("X.size() : {}, y.size() : {}, accumulation_steps : {}".format(X.size(), y.size(), accumulation_steps))
            if isinstance(pred, tuple):
                print("pred[0].size() (mu) : {}, pred[1].size() (var) : {}".format(pred[0].size(), pred[1].size()))
            else:
                print("pred.size() : {}".format(pred.size()))
            raise RuntimeError("Intentional exit")

        last_loss = loss.item()

        # Per-event hit counts vary enormously in this dataset (two orders of magnitude), so
        # consecutive steps request wildly different tensor sizes. The caching allocator doesn't
        # always reuse a block freed by a small event for a subsequent large event's allocation,
        # letting reserved memory climb across steps; releasing it back explicitly keeps peak
        # usage tied to the current step's event rather than the high-water mark of all steps so far.
        del X, y, tau, pred, loss
        if device.type == "cuda":
            torch.cuda.empty_cache()

    return last_loss

def evaluate(
    dataloader : BaseDataLoader,
    model : torch.nn.Module,
    loss_fn : Callable,
    device : torch.device,
    use_amp : bool = False,
    accumulation_steps : int = 1,
    log_every : int = 0,
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
    log_every : int
        Print a one-line progress message every this many steps (batch size, hit count). 0
        disables it. See the matching parameter on train() for why this exists.

    Returns
    -------
    test_loss : float
        Average per-sample loss over all validation events, comparable across runs regardless
        of --batch-size/--hit-budget (loss_fn returns a mean over its own batch, so step/batch
        contributions are reweighted by their sample counts before averaging over all of them)
    """

    model.eval()
    test_loss = 0
    total_samples = 0
    with torch.no_grad():
        for istep in range(dataloader.nsteps):
            X, y, tau = next(dataloader)
            X, y, tau = X.to(device), y.to(device), tau.to(device)
            # BatchedDataLoader batches carry an explicit leading batch dim; UnbatchedDataLoader
            # yields one sample per step with no such dim (matching GaussianNLLLoss's own
            # loss.dim() > 2 check for whether it has a batch dim to sum over).
            total_samples += X.shape[0] if X.dim() > 2 else 1

            if log_every and istep % log_every == 0:
                print(f"[eval] step {istep + 1}/{dataloader.nsteps}: batch_size={X.shape[0]}, nhits={X.shape[-2]}", flush=True)

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
                # loss_fn returns a mean over its own (micro-)batch, so multiplying back by that
                # micro-batch's sample count reconstructs its contribution to the step's total
                # summed loss -- needed to correctly weight micro-batches/steps of different
                # sizes when averaging over the whole validation set below.
                n_sub_samples = X_sub.shape[0] if X_sub.dim() > 2 else 1
                step_loss += micro_loss.item() * n_sub_samples

            test_loss += step_loss

            if not np.isfinite(test_loss):
                print("test_loss:", test_loss)
                print("X:", X)
                print("y:", y)
                print("pred:", pred)
                raise RuntimeError("Nonfinite test loss")

            # See the matching comment in train(): per-event hit counts vary by two orders of
            # magnitude in this dataset, so release each step's tensors explicitly rather than
            # letting the allocator's reserved memory climb across wildly different step sizes.
            del X, y, tau, pred
            if device.type == "cuda":
                torch.cuda.empty_cache()

    # Divide by samples, not steps/batches: test_loss above was reweighted back to per-sample
    # sums (undoing loss_fn's own per-batch mean), so this reconstructs a true overall per-sample
    # average comparable across runs that use different --batch-size/--hit-budget settings.
    return test_loss / total_samples
