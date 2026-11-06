import argparse

import h5py
import numpy as np
import matplotlib.pyplot as plt

def collect_phi(filepaths):
    """
    Walk one or more raw converted HDF5 files (edm4hep_to_hdf5/slcio_to_hdf5 output) and
    compute phi = atan2(y, x) for every hit, regardless of which subdetector collection it
    lives in. Unlike make_training_data.py, no phi-window restriction is applied.

    Returns:
        phi (np.ndarray): phi [-pi, pi] concatenated across all events/collections/files
    """
    all_phi = []

    for filepath in filepaths:
        with h5py.File(filepath, "r") as f:
            x_paths = []
            f.visititems(
                lambda name, obj: x_paths.append(name) if isinstance(obj, h5py.Dataset) and name.endswith("hit_x_pos") else None
            )
            for x_path in x_paths:
                y_path = x_path[:-len("hit_x_pos")] + "hit_y_pos"
                x = f[x_path][:]
                y = f[y_path][:]
                all_phi.append(np.arctan2(y, x))

    return np.concatenate(all_phi) if all_phi else np.array([])

def main(args):
    phi = collect_phi(args.files)

    print(f"{len(phi)} total hits")
    print(f"phi: mean={phi.mean():.3g}, std={phi.std():.3g}, range=[{phi.min():.3g}, {phi.max():.3g}]")

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.hist(phi, bins=args.bins, range=(-np.pi, np.pi), edgecolor="black", alpha=0.7, color="indianred")
    ax.set_xlabel("phi [rad]")
    ax.set_ylabel("Hits")
    ax.set_title("hit phi distribution (unrestricted)")
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(args.out)
    print(f"Wrote {args.out}")

    return 0

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Histogram the full (unrestricted) phi distribution of raw hits from converted HDF5 files")
    parser.add_argument("files", nargs="+", help="One or more raw converted HDF5 files (output of edm4hep_to_hdf5/slcio_to_hdf5)")
    parser.add_argument("-b", "--bins", type=int, default=100, help="Number of histogram bins")
    parser.add_argument("-o", "--out", default="phi_distribution.png", help="Output plot path")
    print("\nFinished with exit code:", main(parser.parse_args()))
