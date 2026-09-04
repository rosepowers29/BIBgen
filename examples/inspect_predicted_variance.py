"""
Inspects a predict_variances=True model's learned variance head as a function of
diffusion timestep tau, without needing real data or running full generation.

Feeds standard-normal synthetic inputs (matching the true forward-process marginal at
tau=T by construction, and used here as a fixed, controlled stress input across ALL tau
so that any tau-dependence observed is attributable to the model's own tau-conditioning,
not to a confound of also varying input realism) and records the model's predicted
variance per feature at each tau.

Useful for diagnosing whether a variance blowup/collapse near specific timesteps (e.g.
tau close to T, where an aggressive schedule like the cosine one has a large single-step
beta) is responsible for a downstream generation defect in one physical variable.
"""
import argparse
import csv

import numpy as np
import torch
import matplotlib.pyplot as plt

from BIBgen.training import load_empty_model

FEATURE_NAMES = ("energy", "phi", "s", "z")

def main(args):
    model_path = args.model_path
    model_config_path = args.model_config
    schedule_path = args.noise_schedule
    outpath = args.out
    assert model_path.endswith(".pth")
    assert model_config_path.endswith(".json")
    assert schedule_path.endswith(".csv")
    assert outpath.endswith(".png")

    device = torch.device("cpu")
    n_timesteps = len(np.loadtxt(schedule_path))

    model = load_empty_model(model_config_path, n_timesteps).to(device)
    model.load_state_dict(torch.load(model_path, weights_only=True, map_location=device))
    model.eval()

    if not model.predict_variances:
        raise ValueError(f"{model_path} was not trained with predict_variances=True -- nothing to inspect")

    taus = torch.arange(n_timesteps)
    mean_var = np.zeros((n_timesteps, 4))
    std_var = np.zeros((n_timesteps, 4))
    max_var = np.zeros((n_timesteps, 4))

    with torch.no_grad():
        for i, tau in enumerate(taus):
            x = torch.randn(args.n_samples, args.n_hits, 4)
            tau_batched = torch.full((args.n_samples,), int(tau), dtype=torch.long)
            _, var = model(x, tau_batched)
            var = var.numpy()
            mean_var[i] = var.mean(axis=(0, 1))
            std_var[i] = var.std(axis=(0, 1))
            max_var[i] = var.max(axis=(0, 1))

    print(f"tau=0 (least noised) mean variance per feature {FEATURE_NAMES}: {mean_var[0]}")
    print(f"tau={n_timesteps - 1} (most noised) mean variance per feature {FEATURE_NAMES}: {mean_var[-1]}")
    print(f"Max variance observed anywhere, per feature: {max_var.max(axis=0)}")

    fig, ax = plt.subplots(figsize=(8, 5))
    for i, name in enumerate(FEATURE_NAMES):
        ax.plot(taus.numpy(), mean_var[:, i], label=name)
        ax.fill_between(taus.numpy(), np.clip(mean_var[:, i] - std_var[:, i], 1e-12, None),
                         mean_var[:, i] + std_var[:, i], alpha=0.15)
    ax.set_yscale("log")
    ax.set_xlabel(r"$\tau$")
    ax.set_ylabel("Predicted variance (whitened units, log scale)")
    ax.set_title(f"Learned variance vs. timestep\n{model_path}")
    ax.legend()
    plt.tight_layout()
    plt.savefig(outpath)
    print(f"Wrote {outpath}")

    csv_path = outpath.rsplit(".", 1)[0] + ".csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["tau"] + [f"{n}_{stat}" for n in FEATURE_NAMES for stat in ("mean", "std", "max")])
        for i, tau in enumerate(taus.tolist()):
            row = [tau]
            for j in range(4):
                row += [mean_var[i, j], std_var[i, j], max_var[i, j]]
            writer.writerow(row)
    print(f"Wrote {csv_path}")

    return 0

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Plot a predict_variances=True model's learned variance vs. tau")
    parser.add_argument("model_path", help="Path to trained model weights .pth file")
    parser.add_argument("model_config", help="json file specifying model name and hyperparameters")
    parser.add_argument("noise_schedule", help="Noise schedule csv used to train this model (fixes n_timesteps)")
    parser.add_argument("-n", "--n-hits", type=int, default=5077, help="Hits per synthetic event (default: mean barrel hits/slice)")
    parser.add_argument("-s", "--n-samples", type=int, default=8, help="Independent synthetic events per tau, for a std estimate")
    parser.add_argument("-o", "--out", default="predicted_variance.png", help="Output plot path (a matching .csv is also written)")
    print("\nFinished with exit code:", main(parser.parse_args()))
