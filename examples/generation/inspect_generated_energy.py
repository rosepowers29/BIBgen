"""
Checks whether the energy channel (feature 0) of a generated .hdf5 has a whitened-space
tail that detonates under np.exp() once unsphered -- the suspected mechanism behind
cosine_predvar's catastrophic Wasserstein blowup, given the reverse-diffusion chain itself
(per examples/generation/run_generate_like_verbose.sh) traces as perfectly well-behaved
when inspected in aggregate across all 4 features.

Pure numpy/h5py -- no torch, no GPU needed, safe to run directly on an access point.
"""
import argparse

import h5py
import numpy as np

def main(args):
    with h5py.File(args.train_file, "r") as f:
        mu = np.array(f["transformation/mu"])
        std = np.array(f["transformation/std"])

    with h5py.File(args.gen_file, "r") as f:
        gen = np.concatenate([np.array(f[k]) for k in f.keys()])

    energy_whitened = gen[:, 0]
    energy_unsphered = std[0] * energy_whitened + mu[0]
    energy_physical = np.exp(energy_unsphered)

    print(f"n hits: {len(energy_whitened)}")
    print("whitened energy:   min={:.4g} mean={:.4g} max={:.4g} p99.9={:.4g}".format(
        energy_whitened.min(), energy_whitened.mean(), energy_whitened.max(),
        np.percentile(energy_whitened, 99.9)))
    print("unsphered logE:     min={:.4g} mean={:.4g} max={:.4g} p99.9={:.4g}".format(
        energy_unsphered.min(), energy_unsphered.mean(), energy_unsphered.max(),
        np.percentile(energy_unsphered, 99.9)))
    print("physical E=exp():  min={:.4g} mean={:.4g} max={:.4g} p99.9={:.4g}".format(
        energy_physical.min(), energy_physical.mean(), energy_physical.max(),
        np.percentile(energy_physical, 99.9)))

    n_extreme = (energy_unsphered > 20).sum()
    print(f"hits with unsphered logE > 20 (exp() > {np.exp(20):.3g}): {n_extreme} / {len(energy_whitened)}")

    return 0

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Diagnose whether generated energy channel detonates under exp()")
    parser.add_argument("gen_file", help="Generated .hdf5 (e.g. cosine_predvar_chaindebug_like.hdf5)")
    parser.add_argument("train_file", help="Training .hdf5 with transformation/mu, transformation/std (e.g. diffused_cyl_phipi4_large_logE_cosine.hdf5)")
    print("\nFinished with exit code:", main(parser.parse_args()))
