from __future__ import annotations

import argparse

import h5py
import numpy as np
import torch

from BIBgen.generation import Generator
from BIBgen.training import load_empty_model

def main(args):
    model_path = args.model_path
    schedule_path = args.noise_schedule
    model_config_path = args.model_config
    cache_path = args.generator_cache
    outpath = args.out
    nevents = args.nevents
    assert model_path.endswith(".pth")
    assert schedule_path.endswith(".csv")
    assert cache_path.endswith(".npz")
    assert outpath.endswith(".hdf5")

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    print(f"Using {device} device")
    schedule = torch.from_numpy(np.loadtxt(schedule_path)).to(device)

    model = load_empty_model(model_config_path, len(schedule)).to(device)
    model.load_state_dict(torch.load(model_path, weights_only=True, map_location=device))
    gen_schedule = None if model.predict_variances else schedule

    generator = Generator()
    generator.load_cache(cache_path)
    generator.load_model(model, device, gen_schedule)

    print(f"Writing generated events to {outpath}")
    with h5py.File(outpath, "w") as fout:
        # Events are written in physical units: un-sphered, with E (not ln E) in GeV.
        # plot_comparison.py reads these attributes to skip its own un-sphering.
        fout.attrs["sphered"] = False
        fout.attrs["log_energy"] = False
        for event_no, event in enumerate(generator.generate(nevents)):
            event_id = "evt_{}".format(event_no)
            fout.create_dataset(event_id, data=event)
            print("Wrote {} of shape {} to {}".format(event_id, event.shape, outpath))

    return 0

# uv run from_cache.py denoiser_v13.pth ../../config/models/v13.json ../../config/noise_schedules/<schedule>.csv generator_cache.npz -o v13_gen.hdf5 -n 1000
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generates events in physical units (E in GeV, phi, s, z), with sizes drawn from the cached distribution of training data")
    parser.add_argument("model_path", help="Path to model weights .pth file")
    parser.add_argument("model_config", help="json file specifying model name and hyperparameters")
    parser.add_argument("noise_schedule", help="Path to noise schedule used in forward diffusion")
    parser.add_argument("generator_cache", help="Path to generator cache written by write_generator_cache.py")
    parser.add_argument("-o", "--out", default="gen.hdf5", help="Path to output events")
    parser.add_argument("-n", "--nevents", default=1000, type=int, help="Number of events to generate")
    print("\nFinished with exit code:", main(parser.parse_args()))
