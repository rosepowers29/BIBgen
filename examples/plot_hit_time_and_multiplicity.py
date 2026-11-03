import argparse

import h5py
import numpy as np
import matplotlib.pyplot as plt

def collect_hit_times_and_counts(filepaths):
    """
    Walk one or more raw converted HDF5 files (edm4hep_to_hdf5/slcio_to_hdf5 output) and
    gather every hit_times dataset found under each event, regardless of which subdetector
    collection it lives in.

    Returns:
        hit_times (np.ndarray): hit_times concatenated across all events/collections/files
        n_hits_per_evt (np.ndarray): total hit count per event, one entry per event across all files
    """
    all_times = []
    n_hits_per_evt = []

    for filepath in filepaths:
        with h5py.File(filepath, "r") as f:
            for evt_id in (k for k in f.keys() if k.startswith("evt_")):
                evt_times = []
                f[evt_id].visititems(
                    lambda name, obj: evt_times.append(obj[:]) if isinstance(obj, h5py.Dataset) and name.endswith("hit_times") else None
                )
                evt_times = np.concatenate(evt_times) if evt_times else np.array([])
                all_times.append(evt_times)
                n_hits_per_evt.append(len(evt_times))

    hit_times = np.concatenate(all_times) if all_times else np.array([])
    return hit_times, np.array(n_hits_per_evt)

def main(args):
    hit_times, n_hits_per_evt = collect_hit_times_and_counts(args.files)

    print(f"{len(n_hits_per_evt)} events, {len(hit_times)} total hits")
    print(f"hit_time: mean={hit_times.mean():.3g}, std={hit_times.std():.3g}, "
          f"range=[{hit_times.min():.3g}, {hit_times.max():.3g}]")
    print(f"n_hits_per_evt: mean={n_hits_per_evt.mean():.3g}, std={n_hits_per_evt.std():.3g}, "
          f"range=[{n_hits_per_evt.min()}, {n_hits_per_evt.max()}]")

    fig, (ax_time, ax_nhits) = plt.subplots(1, 2, figsize=(12, 5))

    ax_time.hist(hit_times, bins=args.bins, range=args.time_range, edgecolor="black", alpha=0.7, color="indianred")
    ax_time.set_xlabel("Hit time [ns]")
    ax_time.set_ylabel("Hits")
    ax_time.set_title("hit_times")
    ax_time.set_yscale("log")
    ax_time.grid(True, alpha=0.3)

    ax_nhits.hist(n_hits_per_evt, bins=args.bins, edgecolor="black", alpha=0.7, color="steelblue")
    ax_nhits.set_xlabel("Hits per event")
    ax_nhits.set_ylabel("Events")
    ax_nhits.set_title("n_hits_per_evt")
    ax_nhits.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(args.out)
    print(f"Wrote {args.out}")

    return 0

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Histogram hit_times and hits-per-event from raw converted HDF5 files")
    parser.add_argument("files", nargs="+", help="One or more raw converted HDF5 files (output of edm4hep_to_hdf5/slcio_to_hdf5)")
    parser.add_argument("-b", "--bins", type=int, default=100, help="Number of histogram bins")
    parser.add_argument("--time-range", nargs=2, type=float, default=None, metavar=("LOW", "HIGH"), help="Optional x-axis range for the hit_time histogram")
    parser.add_argument("-o", "--out", default="hit_time_and_multiplicity.png", help="Output plot path")
    print("\nFinished with exit code:", main(parser.parse_args()))
