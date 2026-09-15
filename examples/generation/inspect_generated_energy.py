"""
Checks whether any feature channel of a generated .hdf5 has a whitened-space tail that
blows up once unsphered -- energy detonates via np.exp() (see BIBgenHistogramAnalyzer.
load_from_model_output), while phi/s/z blow up directly from the linear unsphere step
alone, and eta blows up via -log(tan(theta/2)) if a hit's (s, z) lands near the beam axis.
Useful for isolating which variable(s) are behind a catastrophic Wasserstein-distance
blowup (examples/plot_comparison.py) when the raw reverse-diffusion chain itself
(per examples/generation/run_generate_like_verbose.sh) traces as well-behaved in aggregate.

Pure numpy/h5py -- no torch, no GPU needed, safe to run directly on an access point.
"""
import argparse

import h5py
import numpy as np

FEATURE_NAMES = ("energy", "phi", "s", "z")

def compute_eta(s, z):
    theta = np.abs(np.arctan2(s, z))
    return -np.log(np.tan((theta % (2 * np.pi)) / 2.0 + 1e-10))

def report(name, whitened, unsphered):
    print(f"--- {name} ---")
    print("  whitened:  min={:.4g} mean={:.4g} max={:.4g} p99.9={:.4g}".format(
        whitened.min(), whitened.mean(), whitened.max(), np.percentile(whitened, 99.9)))
    print("  unsphered: min={:.4g} mean={:.4g} max={:.4g} p99.9={:.4g}".format(
        unsphered.min(), unsphered.mean(), unsphered.max(), np.percentile(unsphered, 99.9)))

def main(args):
    with h5py.File(args.train_file, "r") as f:
        mu = np.array(f["transformation/mu"])
        std = np.array(f["transformation/std"])

    with h5py.File(args.gen_file, "r") as f:
        gen = np.concatenate([np.array(f[k]) for k in f.keys()])

    print(f"n hits: {len(gen)}\n")

    unsphered = std * gen + mu

    for i, name in enumerate(FEATURE_NAMES):
        report(name, gen[:, i], unsphered[:, i])

    energy_physical = np.exp(unsphered[:, 0])
    print("--- energy (physical, exp()'d) ---")
    print("  min={:.4g} mean={:.4g} max={:.4g} p99.9={:.4g}".format(
        energy_physical.min(), energy_physical.mean(), energy_physical.max(),
        np.percentile(energy_physical, 99.9)))
    n_extreme_e = (unsphered[:, 0] > 20).sum()
    print(f"  hits with unsphered logE > 20 (exp() > {np.exp(20):.3g}): {n_extreme_e} / {len(gen)}\n")

    s, z = unsphered[:, 2], unsphered[:, 3]
    eta = compute_eta(s, z)
    print("--- eta (derived from unsphered s, z) ---")
    print("  min={:.4g} mean={:.4g} max={:.4g} p99.9={:.4g}".format(
        eta.min(), eta.mean(), eta.max(), np.percentile(eta, 99.9)))
    n_nonfinite_eta = (~np.isfinite(eta)).sum()
    print(f"  non-finite eta values: {n_nonfinite_eta} / {len(gen)}")
    near_axis = (np.abs(s) < 1.0).sum()
    print(f"  hits with |s| < 1.0 (near beam axis, eta blows up here): {near_axis} / {len(gen)}")

    return 0

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Diagnose whether any generated feature channel blows up once unsphered")
    parser.add_argument("gen_file", help="Generated .hdf5 (e.g. cosine_predvar_chaindebug_like.hdf5)")
    parser.add_argument("train_file", help="Training .hdf5 with transformation/mu, transformation/std (e.g. diffused_cyl_phipi4_large_logE_cosine.hdf5)")
    print("\nFinished with exit code:", main(parser.parse_args()))
