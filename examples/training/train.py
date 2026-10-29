import os
import argparse

import h5py
import torch
import numpy as np

from BIBgen.losses import GaussianNLLLoss, DecoupledGaussianNLLLoss
from BIBgen.training import BatchedDataLoader, train, evaluate, load_empty_model
from BIBgen import models

def main(args):
    inpath = args.inpath
    nepochs = args.epochs
    schedule_path = args.noise_schedule
    model_config_path = args.model_config
    batch_size = args.batch_size
    hit_budget = args.hit_budget
    assert inpath.endswith(".hdf5")
    assert schedule_path.endswith(".csv")
    assert model_config_path.endswith(".json")
    if (batch_size is None) == (hit_budget is None):
        raise SystemExit("Specify exactly one of --batch-size or --hit-budget")
    if args.resume and args.init_weights:
        raise SystemExit("--resume and --init-weights are mutually exclusive")

    tag = args.tag or os.path.splitext(os.path.basename(model_config_path))[0]
    out_path = args.out or f"denoiser_{tag}.pth"
    history_path = f"history_{tag}.csv"
    checkpoint_path = f"checkpoint_{tag}.pth"

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    print(f"Using {device} device")

    infile = h5py.File(inpath, "r")
    training_loader = BatchedDataLoader(infile, "training", batch_size=batch_size, hit_budget=hit_budget)
    validation_loader = BatchedDataLoader(infile, "validation", batch_size=batch_size, hit_budget=hit_budget)
    schedule = torch.from_numpy(np.loadtxt(schedule_path)).to(device)

    model = load_empty_model(model_config_path, len(schedule)).to(device)

    if model.predict_variances:
        gaussian_nll = DecoupledGaussianNLLLoss(variance_loss_weight=args.variance_loss_weight)
        loss_fn = lambda pred, y, tau: gaussian_nll(pred[0], pred[1], y)
    else:
        gaussian_nll = GaussianNLLLoss()
        loss_fn = lambda pred, y, tau: gaussian_nll(pred, schedule[tau], y)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-2)
    scaler = torch.amp.GradScaler("cuda") if args.amp and device.type == "cuda" else None

    start_epoch = 0
    best_val_loss = None
    history = None

    if args.resume:
        if os.path.exists(checkpoint_path):
            # Trusted, locally-produced checkpoint (not an external/untrusted file), so the
            # torch>=2.6 weights_only=True default -- which would reject the plain history list
            # and non-tensor optimizer/scaler state below -- doesn't apply here.
            checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
            model.load_state_dict(checkpoint["model"])
            optimizer.load_state_dict(checkpoint["optimizer"])
            if scaler is not None and checkpoint.get("scaler") is not None:
                scaler.load_state_dict(checkpoint["scaler"])
            start_epoch = checkpoint["epoch"] + 1
            best_val_loss = checkpoint["best_val_loss"]
            history = checkpoint["history"]
            print(f"Resumed from {checkpoint_path}: continuing at epoch {start_epoch} (best_val_loss={best_val_loss})")
        else:
            print(f"--resume given but {checkpoint_path} does not exist; starting fresh")

    if history is None and args.init_weights:
        # A plain model.state_dict() (e.g. a prior run's denoiser_<tag>.pth), not a --resume-style
        # checkpoint -- warm-starts the model only. Optimizer/epoch/history all start fresh, which
        # is deliberate: this is for restarting past a derailed run without carrying forward
        # whatever optimizer state (e.g. AdamW momentum) contributed to the derailment.
        model.load_state_dict(torch.load(args.init_weights, map_location=device, weights_only=True))
        print(f"Initialized model weights from {args.init_weights}")

    if history is None:
        best_val_loss = evaluate(validation_loader, model, loss_fn, device, use_amp=args.amp, accumulation_steps=args.accumulation_steps, log_every=args.log_every)
        print("Before training: Validation loss: {}".format(best_val_loss))
        history = [(-1, best_val_loss)]

    if start_epoch >= nepochs:
        print(f"Checkpoint already completed {start_epoch} of {nepochs} epochs; nothing to do")

    for epoch in range(start_epoch, nepochs):
        train(
            training_loader, model, loss_fn, optimizer, device,
            scaler=scaler, accumulation_steps=args.accumulation_steps,
            max_steps_diagnostics=5 if epoch < 5 else 0,
            log_every=args.log_every, max_grad_norm=args.grad_clip,
        )

        if epoch < 10 or epoch % 5 == 0:
            val_loss = evaluate(validation_loader, model, loss_fn, device, use_amp=args.amp, accumulation_steps=args.accumulation_steps, log_every=args.log_every)
            print("Epoch {}: Validation loss: {}".format(epoch, val_loss))
            history.append((epoch, val_loss))
            np.savetxt(history_path, np.array(history), delimiter=",", header="epoch,val_loss", comments="")

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                torch.save(model.state_dict(), out_path)
                print("Saving epoch {} to {}".format(epoch, out_path))

        # Full resumable training state, overwritten every epoch (regardless of whether this
        # epoch ran validation) -- so a job killed by e.g. hitting its walltime loses at most
        # one epoch's progress. Kept separate from out_path, which stays model-weights-only for
        # downstream generation code that doesn't need to know about this format.
        torch.save({
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scaler": scaler.state_dict() if scaler is not None else None,
            "epoch": epoch,
            "best_val_loss": best_val_loss,
            "history": history,
        }, checkpoint_path)

    infile.close()
    return 0

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Script to train a equivariant denoising model")
    parser.add_argument("inpath", help="Diffused training data")
    parser.add_argument("noise_schedule", help="Noise schedule of forward diffusion process")
    parser.add_argument("model_config", help="json file specifying model name and hyperparameters")
    parser.add_argument("-e", "--epochs", type=int, help="Number of epochs to train")
    parser.add_argument("--lr", type=float, default=1e-4, help="AdamW learning rate (default: 1e-4, tuned for the original batch_size=5 runs -- larger batch sizes/hit budgets mean far fewer gradient updates per epoch, so may need a proportionally higher value)")
    parser.add_argument("-b", "--batch-size", type=int, help="Fixed batch size, the same for every event. Mutually exclusive with --hit-budget")
    parser.add_argument("--hit-budget", type=int, help="Instead of a fixed batch size, cap each batch at roughly this many total hits (batch_size x n_hits), with batch_size chosen per-event as the largest divisor of the schedule's timestep count that fits. Use this when per-event hit counts vary enormously, so a single fixed --batch-size either OOMs on the largest events or wastes GPU utilization on the rest. Mutually exclusive with --batch-size")
    parser.add_argument("--variance-loss-weight", type=float, default=1.0, help="Weight on the variance-training loss term (only used when predict_variances=True)")
    parser.add_argument("-o", "--out", default=None, help="Output .pth path (default: denoiser_<tag>.pth)")
    parser.add_argument("-t", "--tag", default=None, help="Experiment tag for naming outputs (default: config filename stem)")
    parser.add_argument("--amp", action="store_true", help="Train under automatic mixed precision (torch.autocast + GradScaler) to reduce GPU memory use")
    parser.add_argument("--accumulation-steps", type=int, default=1, help="Split each batch into this many micro-batches, accumulating gradients before each optimizer step, to reduce peak GPU memory use without changing batch size (must divide batch size evenly)")
    parser.add_argument("--log-every", type=int, default=0, help="Print a one-line progress message (batch size, hit count) every this many steps, during both training and evaluation. 0 (default) disables it -- meant for watching a run live (e.g. an interactive node), not for long batch jobs where it would just bloat the log")
    parser.add_argument("--resume", action="store_true", help="Resume from checkpoint_<tag>.pth (model, optimizer, scaler, epoch, and history) if it exists, continuing to -e/--epochs. If no checkpoint is found, starts fresh. Use this to continue a run that was killed (e.g. hit its walltime) without losing progress")
    parser.add_argument("--init-weights", default=None, help="Warm-start the model from a plain model.state_dict() .pth file (e.g. a prior run's denoiser_<tag>.pth), with a fresh optimizer/epoch-0/history -- unlike --resume, does not carry over optimizer state. Use this to restart past a derailed run from an earlier good checkpoint without inheriting whatever optimizer state caused the derailment. Mutually exclusive with --resume")
    parser.add_argument("--grad-clip", type=float, default=None, help="Clip each step's global gradient norm to this value before the optimizer step, to guard against a single pathological batch derailing training for many subsequent epochs. None (default) disables clipping")
    print("\nFinished with exit code:", main(parser.parse_args()))
