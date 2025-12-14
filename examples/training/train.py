import argparse

import h5py
import torch
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument("inpath")
parser.add_argument("-c", "--condor", action="store_true")
parser.add_argument("-e", "--epochs", type=int)
args = parser.parse_args()

if args.condor:
    from diffusion import EquivariantDenoiser
    from gaussian_nll import GaussianNLLLoss
    from training import DataLoader, train, evaluate
else:
    from BIBgen.models import EquivariantDenoiser
    from BIBgen.loss_function import GaussianNLLLoss
    from BIBgen.training import DataLoader, train, evaluate

def main(args):
    inpath = args.inpath
    nepochs = args.epochs
    assert inpath.endswith(".hdf5")

    # device = torch.accelerator.current_accelerator().type if torch.accelerator.is_available() else "cpu"
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    print(f"Using {device} device")

    infile = h5py.File(inpath, "r")
    training_loader = DataLoader(infile, "training")
    validation_loader = DataLoader(infile, "validation")

    model = EquivariantDenoiser(
        n_timesteps = 100,
        tau_encoding_dimension = 32,
        position_encoding_dimension = 64,
        hidden_layer_size = 256,
        n_hidden_layers = 4
    ).to(device)
    # model = EquivariantDenoiser(
    #     n_timesteps = 100,
    #     tau_encoding_dimension = 8,
    #     position_encoding_dimension = 16,
    #     hidden_layer_size = 32,
    #     n_hidden_layers = 1
    # )

    gaussian_nll = GaussianNLLLoss()
    loss_fn = lambda pred, y: gaussian_nll(*pred, y)

    optimizer = torch.optim.SGD(model.parameters(), lr=1e-7)
    
    best_val_loss = np.inf
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