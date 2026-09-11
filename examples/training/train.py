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
    assert inpath.endswith(".hdf5")
    assert schedule_path.endswith(".csv")
    assert model_config_path.endswith(".json")

    tag = args.tag or os.path.splitext(os.path.basename(model_config_path))[0]
    out_path = args.out or f"denoiser_{tag}.pth"
    history_path = f"history_{tag}.csv"

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    print(f"Using {device} device")

    infile = h5py.File(inpath, "r")
    training_loader = BatchedDataLoader(infile, "training", batch_size=batch_size)
    validation_loader = BatchedDataLoader(infile, "validation", batch_size=batch_size)
    schedule = torch.from_numpy(np.loadtxt(schedule_path)).to(device)

    model = load_empty_model(model_config_path, len(schedule)).to(device)

    if model.predict_variances:
        gaussian_nll = DecoupledGaussianNLLLoss(variance_loss_weight=args.variance_loss_weight)
        loss_fn = lambda pred, y, tau: gaussian_nll(pred[0], pred[1], y)
    else:
        gaussian_nll = GaussianNLLLoss()
        loss_fn = lambda pred, y, tau: gaussian_nll(pred, schedule[tau], y)

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-2)

    best_val_loss = evaluate(validation_loader, model, loss_fn, device)
    print("Before training: Validation loss: {}".format(best_val_loss))

    history = [(-1, best_val_loss)]

    for epoch in range(nepochs):
        train(training_loader, model, loss_fn, optimizer, device, max_steps_diagnostics=5 if epoch < 5 else 0)

        if epoch < 10 or epoch % 5 == 0:
            val_loss = evaluate(validation_loader, model, loss_fn, device)
            print("Epoch {}: Validation loss: {}".format(epoch, val_loss))
            history.append((epoch, val_loss))
            np.savetxt(history_path, np.array(history), delimiter=",", header="epoch,val_loss", comments="")

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                torch.save(model.state_dict(), out_path)
                print("Saving epoch {} to {}".format(epoch, out_path))

    infile.close()
    return 0

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Script to train a equivariant denoising model")
    parser.add_argument("inpath", help="Diffused training data")
    parser.add_argument("noise_schedule", help="Noise schedule of forward diffusion process")
    parser.add_argument("model_config", help="json file specifying model name and hyperparameters")
    parser.add_argument("-e", "--epochs", type=int, help="Number of epochs to train")
    parser.add_argument("-b", "--batch-size", type=int, help="Batch size")
    parser.add_argument("--variance-loss-weight", type=float, default=1.0, help="Weight on the variance-training loss term (only used when predict_variances=True)")
    parser.add_argument("-o", "--out", default=None, help="Output .pth path (default: denoiser_<tag>.pth)")
    parser.add_argument("-t", "--tag", default=None, help="Experiment tag for naming outputs (default: config filename stem)")
    print("\nFinished with exit code:", main(parser.parse_args()))
