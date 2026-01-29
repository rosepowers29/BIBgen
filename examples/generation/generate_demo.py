import argparse

import os
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"

import h5py
import torch
import numpy as np

parser = argparse.ArgumentParser(description="Generates demo events with a trained model with a specific number of hits")
parser.add_argument("model_path")
parser.add_argument("noise_schedule")
parser.add_argument("-c", "--condor", action="store_true", help="Script is running in a condor job. Will import from local files.")
parser.add_argument("-n", "--nevents", type=int, help="Number of events to generate")
parser.add_argument("-s", "--size", type=int, help="Number of hits per event")
args = parser.parse_args()

if args.condor:
    from diffusion import EquivariantDenoiser
    from generation import generate_sphered
else:
    from BIBgen.models import EquivariantDenoiser
    from BIBgen.generation import generate_sphered

def main(args):
    model_path = args.model_path
    nhits = args.size
    nevents = args.nevents
    schedule_path = args.noise_schedule
    assert model_path.endswith(".pth")
    assert schedule_path.endswith(".csv")

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    print(f"Using {device} device")

    schedule = torch.from_numpy(np.loadtxt(schedule_path)).to(device)
    
    model = EquivariantDenoiser(
        n_timesteps = 100,
        tau_encoding_dimension = 32,
        position_encoding_dimension = 64,
        hidden_layer_size = 256,
        n_hidden_layers = 4
    )
    model.load_state_dict(torch.load(model_path, weights_only=True))
    
    with h5py.File("demo.hdf5", "w") as fout:
        for event_no in range(nevents):
            sphered = generate_sphered(model, nhits, device, schedule=schedule, demo=False).cpu()
            fout.create_dataset("evt_{}".format(event_no), data=sphered)
            print("Wrote evt_{} of shape {} to demo.hdf5".format(event_no, sphered.size()))

    return 0

if __name__ == "__main__":
    print("\nFinished with exit code:", main(args))