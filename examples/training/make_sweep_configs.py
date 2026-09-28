"""
Generate a grid of model configs for a Fourier position-encoding scale sweep
(Tancik et al. Fig. 4/9 style), and append matching rows to a Condor
experiments manifest.

Each config is a copy of a base config (e.g. ../../config/equivariant_denoiser.json)
with use_position_encoding/position_encoding_kind/position_encoding_scale set for
one (sigma_phi, sigma_s, sigma_z) triple. Configs are written under
../../config/sweep/, and one row per config is appended to
experiments_fourier_sweep.txt in the same data,config,tag,schedule,loss format
already used by experiments.txt / submit_train.sub.

For "gaussian", --seed additionally reruns each (sigma_phi, sigma_s, sigma_z)
triple once per given seed value (via position_encoding_seed), so a tight/tied
result can be checked for robustness against the random frequency draw rather
than trusting a single uncontrolled sample. Ignored for "positional", which has
no randomness to seed.

Example
-------
uv run make_sweep_configs.py gaussian --phi 0.5 1 2 4 8 16 32 64 --s 1 --z 1 \\
    --data diffused_cyl_phipi4_large_logE.hdf5 --schedule noise_schedule.csv

uv run make_sweep_configs.py gaussian --phi 4 --s 2 4 --z 1 --seed 0 1 2 \\
    --data diffused_cyl_phipi4_large_logE.hdf5 --schedule noise_schedule.csv \\
    --manifest experiments_fourier_sweep_phaseB_reseed.txt
"""
import argparse
import itertools
import json
import os

CONFIG_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "config")
SWEEP_DIR = os.path.join(CONFIG_DIR, "sweep")


def format_scale(value):
    return f"{value:g}".replace(".", "p").replace("-", "neg")


def make_tag(kind, sigma_phi, sigma_s, sigma_z, seed=None):
    tag = f"{kind}_phi{format_scale(sigma_phi)}_s{format_scale(sigma_s)}_z{format_scale(sigma_z)}"
    if seed is not None:
        tag += f"_seed{seed}"
    return tag


def main(args):
    base_config_path = os.path.join(CONFIG_DIR, args.base_config)
    with open(base_config_path) as fin:
        base_config = json.load(fin)

    os.makedirs(SWEEP_DIR, exist_ok=True)
    manifest_path = os.path.join(os.path.dirname(__file__), args.manifest)

    seeds = args.seed if args.seed else [None]

    rows = []
    for sigma_phi, sigma_s, sigma_z, seed in itertools.product(args.phi, args.s, args.z, seeds):
        tag = make_tag(args.kind, sigma_phi, sigma_s, sigma_z, seed)

        config = json.loads(json.dumps(base_config))  # deep copy
        config["hyperparameters"]["use_position_encoding"] = True
        config["hyperparameters"]["position_encoding_kind"] = args.kind
        config["hyperparameters"]["position_encoding_scale"] = [sigma_phi, sigma_s, sigma_z]
        if seed is not None:
            config["hyperparameters"]["position_encoding_seed"] = seed

        config_filename = f"{tag}.json"
        with open(os.path.join(SWEEP_DIR, config_filename), "w") as fout:
            json.dump(config, fout, indent=2)

        rows.append(",".join([args.data, f"sweep/{config_filename}", tag, args.schedule, args.loss]))
        print(f"Wrote config/sweep/{config_filename} (tag={tag})")

    with open(manifest_path, "a") as fout:
        fout.write("\n".join(rows) + "\n")
    print(f"Appended {len(rows)} rows to {manifest_path}")

    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("kind", choices=["positional", "gaussian"], help="Fourier feature strategy to sweep")
    parser.add_argument("--base-config", default="equivariant_denoiser.json", help="Config (relative to ../../config) to copy hyperparameters from")
    parser.add_argument("--phi", type=float, nargs="+", required=True, help="sigma_phi values to sweep (or a single fixed value)")
    parser.add_argument("--s", type=float, nargs="+", required=True, help="sigma_s values to sweep (or a single fixed value)")
    parser.add_argument("--z", type=float, nargs="+", required=True, help="sigma_z values to sweep (or a single fixed value)")
    parser.add_argument("--seed", type=int, nargs="+", default=None, help="Reseed each (phi, s, z) triple once per seed (only meaningful for --kind gaussian); omit for the default single uncontrolled draw per triple")
    parser.add_argument("--data", required=True, help="Diffused training data filename (as used in experiments.txt)")
    parser.add_argument("--schedule", default="noise_schedule.csv", help="Noise schedule filename (relative to config/)")
    parser.add_argument("--loss", default="simple", choices=["simple", "nelbo"])
    parser.add_argument("--manifest", default="experiments_fourier_sweep.txt", help="Manifest file to append rows to")
    print("\nFinished with exit code:", main(parser.parse_args()))
