import os
import json
import argparse

import h5py
import torch
import numpy as np

from BIBgen.losses import GaussianNLLLoss, DecoupledGaussianNLLLoss
from BIBgen.training import BatchedDataLoader, hyperoptimize

def main(args):
    inpath = args.inpath
    nepochs = args.epochs
    schedule_path = args.noise_schedule
    model_template_config_path = args.model_template_config
    batch_size = args.batch_size
    outpath = args.out
    ntrials = args.trials
    assert inpath.endswith(".hdf5")
    assert schedule_path.endswith(".csv")
    assert model_template_config_path.endswith(".json")
    assert outpath.endswith(".json")

    # The study is saved next to the output config under the same name, e.g. best.json -> best.db,
    # and resumed from there if it already exists.
    study_path = os.path.splitext(outpath)[0] + ".db"
    study_name = os.path.basename(os.path.splitext(outpath)[0])
    os.makedirs(os.path.dirname(outpath) or ".", exist_ok=True)
    print(f"Study {study_name} saved in {study_path}" + (" (resuming)" if os.path.exists(study_path) else ""))

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    print(f"Using {device} device")

    with open(model_template_config_path, "r") as fin:
        model_template = json.load(fin)

    schedule = torch.from_numpy(np.loadtxt(schedule_path)).to(device)

    with h5py.File(inpath, "r") as infile:
        phi_std = float(infile["transformation/std"][1]) if args.cylindrical else None

        if "predict_variances" in model_template["hyperparameters"] and model_template["hyperparameters"]["predict_variances"]:
            gaussian_nll = DecoupledGaussianNLLLoss(variance_loss_weight=args.variance_loss_weight, circular_std=phi_std)
            loss_fn = lambda pred, y, tau: gaussian_nll(pred[0], pred[1], y)
        else:
            gaussian_nll = GaussianNLLLoss(circular_std=phi_std)
            loss_fn = lambda pred, y, tau: gaussian_nll(pred, schedule[tau], y)

        training_loader = BatchedDataLoader(infile, "training", batch_size=batch_size)
        validation_loader = BatchedDataLoader(infile, "validation", batch_size=batch_size)

        best_val_loss, best_config, ntrials_done = hyperoptimize(
            model_template,
            training_loader, 
            validation_loader, 
            loss_fn,
            device,
            n_timesteps=len(schedule),
            nepochs=nepochs,
            ntrials=ntrials,
            storage="sqlite:///" + os.path.abspath(study_path),
            study_name=study_name,
            timeout=args.timeout * 3600,
        )

    print(f"{ntrials_done} of {ntrials} trials done")
    print("Best validation loss:", best_val_loss)
    if ntrials_done < ntrials:
        print(f"Study not finished; run again with the same arguments to resume it from {study_path}")
        return 0
    if best_config is None:
        print("No trial completed, so there is no best config to write")
        return 1

    with open(outpath, "w") as fout:
        json.dump(best_config, fout, indent=4)
    print("Wrote best config to", outpath)
    return 0

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Script to hyperoptimize an equivariant denoising model")
    parser.add_argument("inpath", help="Diffused training data")
    parser.add_argument("noise_schedule", help="Noise schedule of forward diffusion process")
    parser.add_argument("model_template_config", help="json file specifying model name and hyperparameter ranges")
    parser.add_argument("-e", "--epochs", type=int, default=20, help="Number of epochs to train each trial")
    parser.add_argument("-n", "--trials", type=int, default=40, help="Total number of trials, summed over every run that resumes the study")
    parser.add_argument("--timeout", type=float, default=20.0, help="Hours after which no new trial is started; the trial in progress still finishes")
    parser.add_argument("-b", "--batch-size", type=int, default=5, help="Batch size")
    parser.add_argument("--variance-loss-weight", type=float, default=1.0, help="Weight on the variance-training loss term (only used when predict_variances=True)")
    parser.add_argument("-o", "--out", default="best.json", help="Output optimal model config, written once all trials are done. The study is saved alongside it as a .db of the same name")
    parser.add_argument("--cylindrical", action="store_true", help="Treat feature index 1 (phi) as a wrapped angle rather than an unbounded coordinate, so the loss measures phi differences the short way around the circle (e.g. 2*pi - 0.1 and 0.1 are 0.2 apart, not 2*pi - 0.2). Only correct for training data produced with make_training_data.py --cylindrical; has no effect on cartesian-coordinate data")
    print("\nFinished with exit code:", main(parser.parse_args()))