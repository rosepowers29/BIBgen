"""
Generates tiny inputs for pre-submit debugging of the condor generation pipeline
(argument mapping, file transfer, python environment) without waiting on a GPU slot.

Produces:
  - ../training/denoiser_debug.pth              (untrained but architecturally-valid
                                                  checkpoint, matching
                                                  config/vsmall_equivariant_denoiser.json
                                                  and config/debug_noise_schedule.csv's 10
                                                  steps -- only needs to load correctly,
                                                  not be accurate)
  - debug_fixtures/test_sizes_large.csv         (a few tiny synthetic event sizes, named
                                                  to match run_generate_like.sh's hardcoded
                                                  size-file argument -- kept in its own
                                                  subdirectory so it can never collide with
                                                  a real test_sizes_large.csv sitting next
                                                  to submit_generate_like.sub; condor's
                                                  transfer_input_files lands it by basename
                                                  regardless of local subdirectory)

Run this from examples/generation/ so everything lands where submit_debug_generate.sub expects it.
"""
import os

import numpy as np
import torch

from BIBgen.models import EquivariantDenoiser

NTAU = 10
NEVENTS = 3
NHITS = 10

def main():
    model = EquivariantDenoiser(
        n_timesteps=NTAU,
        tau_encoding_dimension=4,
        position_encoding_dimension=8,
        hidden_layer_size=16,
        n_hidden_layers=1,
    )
    os.makedirs("../training", exist_ok=True)
    torch.save(model.state_dict(), "../training/denoiser_debug.pth")

    os.makedirs("debug_fixtures", exist_ok=True)
    sizes = [["evt_{}".format(i), NHITS] for i in range(NEVENTS)]
    np.savetxt("debug_fixtures/test_sizes_large.csv", sizes, delimiter=",", fmt="%s")

    print("Wrote ../training/denoiser_debug.pth and debug_fixtures/test_sizes_large.csv")

if __name__ == "__main__":
    main()
