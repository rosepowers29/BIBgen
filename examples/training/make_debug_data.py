"""
Generates a tiny synthetic diffused dataset for pre-submit debugging of the condor
training pipeline (argument mapping, file transfer, python environment) without waiting
on a GPU slot or the real, large, osdf-hosted dataset.

Pairs with config/debug_noise_schedule.csv (10 steps) and config/vsmall_equivariant_denoiser.json.
Run this from examples/training/ so debug_diffused.hdf5 lands next to submit_debug_train.sub.
"""
import h5py
import numpy as np

NTAU = 10
NHITS = 10
NEVENTS = 2

def main():
    with h5py.File("debug_diffused.hdf5", "w") as fout:
        for split in ("training", "validation"):
            group = fout.create_group(split)
            for event_no in range(NEVENTS):
                pseudodata = np.random.rand(NTAU + 1, NHITS, 4).astype(np.float32)
                group.create_dataset("evt_{}".format(event_no), data=pseudodata)

    print("Wrote debug_diffused.hdf5 ({} events x {} timesteps x {} hits)".format(NEVENTS, NTAU, NHITS))

if __name__ == "__main__":
    main()
