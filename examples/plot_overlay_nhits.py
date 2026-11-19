import argparse

import h5py
import numpy as np
import matplotlib.pyplot as plt

SPLITS = ("training", "validation", "test")

def read_nhits(filepath):
    """
    Read the number of hits in every event of a make_training_data.py output file.

    Events are keyed as "<split>/<evt_id>" since evt ids are unique across splits but this
    keeps the provenance explicit. Only dataset shapes are read, not the hit data itself.

    Returns:
        nhits (dict): {"<split>/<evt_id>": number of hits}
    """
    nhits = {}
    with h5py.File(filepath, "r") as f:
        for split in SPLITS:
            if split not in f:
                continue
            for evt_id, evt_group in f[split].items():
                nhits[f"{split}/{evt_id}"] = evt_group["tau0"].shape[0]
    return nhits

def sample_overlays(nhits, n_per_overlay, n_samples, replace, rng):
    """
    Build n_samples pseudo-overlays, each the sum of nhits over n_per_overlay randomly chosen events.

    Returns:
        overlay_nhits (np.ndarray): total hits per pseudo-overlay, shape (n_samples,)
    """
    counts = np.fromiter(nhits.values(), dtype=np.int64, count=len(nhits))
    if not replace and n_per_overlay > len(counts):
        raise ValueError(f"Cannot draw {n_per_overlay} events without replacement from {len(counts)} events")
    return np.array([
        counts[rng.choice(len(counts), size=n_per_overlay, replace=replace)].sum()
        for _ in range(n_samples)
    ])

def main(args):
    nhits = read_nhits(args.file)
    counts = np.array(list(nhits.values()))
    print(f"Read {len(nhits)} events from {args.file}")
    print(f"n_hits per event: mean={counts.mean():.4g}, std={counts.std():.4g}, "
          f"range=[{counts.min()}, {counts.max()}]")

    rng = np.random.default_rng(args.seed)
    overlay_nhits = sample_overlays(nhits, args.n_per_overlay, args.n_samples, args.replace, rng)

    print(f"{args.n_samples} overlays of {args.n_per_overlay} events "
          f"({'with' if args.replace else 'without'} replacement):")
    print(f"  n_hits per overlay: mean={overlay_nhits.mean():.6g}, std={overlay_nhits.std(ddof=1):.4g}, "
          f"range=[{overlay_nhits.min()}, {overlay_nhits.max()}]")
    print(f"  naive expectation: mean={args.n_per_overlay * counts.mean():.6g}")

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.hist(overlay_nhits, bins=args.bins, edgecolor="black", alpha=0.7, color="steelblue")
    ax.axvline(overlay_nhits.mean(), color="black", linestyle="--", label=f"mean = {overlay_nhits.mean():.4g}")
    ax.set_xlabel(f"Hits per overlay ({args.n_per_overlay} events)")
    ax.set_ylabel("Samples")
    ax.set_title(f"n_hits per overlay, {args.n_samples} random samples")
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(args.out)
    print(f"Wrote {args.out}")

    return 0

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Estimate the n_hits distribution of a full BIB overlay by resampling events from a make_training_data.py output file")
    parser.add_argument("file", help="Processed HDF5 file (output of make_training_data.py)")
    parser.add_argument("-k", "--n-per-overlay", type=int, default=1666, help="Number of events summed into one overlay")
    parser.add_argument("-n", "--n-samples", type=int, default=100, help="Number of random overlays to draw")
    parser.add_argument("--replace", action="store_true", help="Draw events within an overlay with replacement (default: without)")
    parser.add_argument("--seed", type=int, default=None, help="RNG seed for reproducibility")
    parser.add_argument("-b", "--bins", type=int, default=20, help="Number of histogram bins")
    parser.add_argument("-o", "--out", default="overlay_nhits.png", help="Output plot path")
    print("\nFinished with exit code:", main(parser.parse_args()))
