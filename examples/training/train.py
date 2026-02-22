import argparse

import h5py
import torch
import numpy as np

parser = argparse.ArgumentParser(description="Script to train a equivariant denoising model")
parser.add_argument("inpath", help="Diffused training data")
parser.add_argument("noise_schedule", help="Noise schedule of forward diffusion process")
parser.add_argument("-c", "--condor", action="store_true", help="Script is running in a condor job. Affects import paths.")
parser.add_argument("-e", "--epochs", type=int, help="Number of epochs to train")
parser.add_argument("-b", "--batch-size", type=int, help="Batch size")
args = parser.parse_args()

if args.condor:
    from diffusion import EquivariantDenoiser
    from losses import GaussianNLLLoss
    from training import BatchedDataLoader, train, evaluate
else:
    from BIBgen.models import EquivariantDenoiser
    from BIBgen.losses import GaussianNLLLoss
    from BIBgen.training import BatchedDataLoader, train, evaluate

def main(args):
    inpath = args.inpath
    nepochs = args.epochs
    schedule_path = args.noise_schedule
    batch_size = args.batch_size
    assert inpath.endswith(".hdf5")
    assert schedule_path.endswith(".csv")

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    print(f"Using {device} device")

    infile = h5py.File(inpath, "r")
    training_loader = BatchedDataLoader(infile, "training", batch_size=batch_size)
    validation_loader = BatchedDataLoader(infile, "validation", batch_size=batch_size)

    model = EquivariantDenoiser(
        n_timesteps = training_loader.ntau,
        tau_encoding_dimension = 32,
        position_encoding_dimension = 64,
        hidden_layer_size = 256,
        n_hidden_layers = 4
    ).to(device)

    schedule = torch.from_numpy(np.loadtxt(schedule_path)).to(device)
    gaussian_nll = GaussianNLLLoss()
    loss_fn = lambda pred, y, tau: gaussian_nll(pred, schedule[tau], y)

    optimizer = torch.optim.SGD(model.parameters(), lr=1e-9)
    # scaler = torch.amp.GradScaler()

    best_val_loss = evaluate(validation_loader, model, loss_fn, device)
    print("Before training: Validation loss: {}".format(best_val_loss))

    for epoch in range(nepochs):
        train(training_loader, model, loss_fn, optimizer, device)

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
    print("\nFinished with exit code:", main(args))
