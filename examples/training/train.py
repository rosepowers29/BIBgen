import argparse
import json

import h5py
import torch
import numpy as np

from BIBgen.losses import GaussianNLLLoss
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

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    print(f"Using {device} device")

    infile = h5py.File(inpath, "r")
    training_loader = BatchedDataLoader(infile, "training", batch_size=batch_size)
    validation_loader = BatchedDataLoader(infile, "validation", batch_size=batch_size)
    schedule = torch.from_numpy(np.loadtxt(schedule_path)).to(device)

    model = load_empty_model(model_config_path, len(schedule)).to(device)

    gaussian_nll = GaussianNLLLoss()
    loss_fn = lambda pred, y, tau: gaussian_nll(pred, schedule[tau], y)

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-2)
    # scaler = torch.amp.GradScaler()

    best_val_loss = evaluate(validation_loader, model, loss_fn, device)
    print("Before training: Validation loss: {}".format(best_val_loss))

    for epoch in range(nepochs):
        train(training_loader, model, loss_fn, optimizer, device, max_steps_diagnostics=5 if epoch < 5 else 0)

        if epoch < 10 or epoch % 5 == 0:
            val_loss = evaluate(validation_loader, model, loss_fn, device)
            print("Epoch {}: Validation loss: {}".format(epoch, val_loss))

            if val_loss < best_val_loss:
                best_val_loss = val_loss

                torch.save(model.state_dict(), "denoiser.pth")
                print("Saving epoch {}".format(epoch))

    infile.close()
    return 0

if __name__ == "__main__":
    # Testing: uv run train.py ../../data/diffused_cyl_phipi4_medium.hdf5 ../../config/noise_schedule.csv ../../config/vsmall_equivariant_denoiser.json -e 1 -b 5
    parser = argparse.ArgumentParser(description="Script to train a equivariant denoising model")
    parser.add_argument("inpath", help="Diffused training data")
    parser.add_argument("noise_schedule", help="Noise schedule of forward diffusion process")
    parser.add_argument("model_config", help="json file specifying model name and hyperparameters")
    parser.add_argument("-e", "--epochs", type=int, help="Number of epochs to train")
    parser.add_argument("-b", "--batch-size", type=int, help="Batch size")
    print("\nFinished with exit code:", main(parser.parse_args()))
