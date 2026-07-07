import argparse
import os
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"

import h5py
import torch
import numpy as np

from BIBgen.models import EquivariantDenoiser
from BIBgen.generation import generate_sphered
from BIBgen.training import load_empty_model

def main(args):
    model_path = args.model_path
    schedule_path = args.noise_schedule
    size_path = args.size_file
    outpath = args.out
    model_config_path = args.model_config
    assert model_path.endswith(".pth")
    assert schedule_path.endswith(".csv")
    assert size_path.endswith(".csv")
    assert outpath.endswith(".hdf5")

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    print(f"Using {device} device")

    schedule = torch.from_numpy(np.loadtxt(schedule_path)).to(device)
    test_event_ids, sizes = np.loadtxt(size_path, delimiter=",", unpack=True, dtype=str)
    sizes = sizes.astype(int)
    
    model = load_empty_model(model_config_path, len(schedule)).to(device)
    model.load_state_dict(torch.load(model_path, weights_only=True))
    
    with h5py.File(outpath, "w") as fout:
        for event_id, nhits in zip(test_event_ids, sizes):
            sphered = generate_sphered(model, nhits, device, schedule=schedule, demo=False).cpu()
            fout.create_dataset(event_id, data=sphered)
            print("Wrote {} of shape {} to demo.hdf5".format(event_id, sphered.size()))

    return 0

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generates demo events with a trained model of sizes provided by size_file")
    parser.add_argument("model_path", help="Path to model weights .pth file")
    parser.add_argument("model_config", help="json file specifying model name and hyperparameters")
    parser.add_argument("noise_schedule", help="Path to noise schedule used in foward diffusion")
    parser.add_argument("size_file", help="Path to file containing sizes of events to generate")
    parser.add_argument("-o", "--out", default="like.hdf5", help="Path to output generate events")
    print("\nFinished with exit code:", main(parser.parse_args()))
