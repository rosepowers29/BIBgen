import argparse
import warnings

import h5py
import numpy as np

from BIBgen.preprocessing import Sphering

SPLITS = ("training", "validation", "test")
PHI = 1  # feature index of phi in cylindrical make_training_data.py output: [E, phi, s, z, t]

def load_unsphered_events(infile):
    """
    Load every event of a make_training_data.py output file back into unsphered coordinates.
    training/validation were sphered with the file's transformation; test is stored unsphered.

    Returns:
        events (list[np.ndarray]): one (nhits, nfeatures) array per source event
    """
    sphering = Sphering(np.array(infile["transformation/mu"]), np.array(infile["transformation/std"]))
    events = []
    for split in SPLITS:
        if split not in infile:
            continue
        for evt_id in infile[split].keys():
            data = np.array(infile[f"{split}/{evt_id}/tau0"])
            events.append(data if split == "test" else sphering.untransform(data))
    return events

def make_slice_edges(events, n_slices, mode, phi_window):
    """
    Phi slice edges spanning [-phi_window, phi_window].
    "width": equal-width slices. "population": edges at quantiles of the pooled hit phi,
    so every slice has the same expected nhits.
    """
    if mode == "width":
        return np.linspace(-phi_window, phi_window, n_slices + 1)
    phi = np.concatenate([e[:, PHI] for e in events])
    edges = np.quantile(phi, np.linspace(0, 1, n_slices + 1))
    edges[0], edges[-1] = -phi_window, phi_window
    return edges

def assign_slices_to_splits(n_slices, fractions):
    """
    Interleave slices across splits so each split is spread around phi rather than in one
    contiguous block: each slice goes to the split furthest below its target share so far.

    Returns:
        split_of_slice (np.ndarray): split index (into SPLITS) for each slice
    """
    fractions = np.asarray(fractions, dtype=float) / np.sum(fractions)
    counts = np.zeros(len(fractions))
    split_of_slice = np.empty(n_slices, dtype=np.int64)
    for i in range(n_slices):
        isplit = np.argmax(fractions * (i + 1) - counts)
        split_of_slice[i] = isplit
        counts[isplit] += 1
    return split_of_slice

def bin_event_by_slice(event, edges):
    """Split one event's hits into a list of per-slice arrays."""
    n_slices = len(edges) - 1
    slice_ids = np.clip(np.searchsorted(edges, event[:, PHI], side="right") - 1, 0, n_slices - 1)
    order = np.argsort(slice_ids, kind="stable")
    bounds = np.cumsum(np.bincount(slice_ids, minlength=n_slices))[:-1]
    return np.split(event[order], bounds)

def main(args):
    inpath = args.infile
    outpath = args.out
    fractions = [float(x) for x in args.split.split(",")]
    assert inpath.endswith(".hdf5")
    assert outpath.endswith(".hdf5")
    assert len(fractions) == 3

    rng = np.random.default_rng(args.seed)

    with h5py.File(inpath, "r") as infile:
        events = load_unsphered_events(infile)
        log_energy = bool(infile["transformation"].attrs.get("log_energy", False))
        log_time = bool(infile["transformation"].attrs.get("log_time", False))
    print(f"Loaded {len(events)} source events from {inpath}")

    max_abs_phi = max(np.abs(e[:, PHI]).max() for e in events if len(e))
    if max_abs_phi > np.pi + 1e-6:
        raise ValueError(f"Feature {PHI} reaches |{max_abs_phi:.3g}| > pi; input must be made with make_training_data.py --cylindrical")
    if max_abs_phi > args.phi_window + 1e-6:
        raise ValueError(f"Data extends to |phi| = {max_abs_phi:.4g}, beyond --phi-window {args.phi_window:.4g}")
    if max_abs_phi < 0.9 * args.phi_window:
        warnings.warn(f"Data only reaches |phi| = {max_abs_phi:.4g} but --phi-window is {args.phi_window:.4g}; "
                      "was the input made with a narrower -p? Outer slices will be sparse.")
    if not args.replace and args.n_per_overlay > len(events):
        raise ValueError(f"Cannot overlay {args.n_per_overlay} events without replacement from {len(events)}")

    edges = make_slice_edges(events, args.n_slices, args.slice_mode, args.phi_window)
    split_of_slice = assign_slices_to_splits(args.n_slices, fractions)
    for isplit, split in enumerate(SPLITS):
        if fractions[isplit] > 0 and not np.any(split_of_slice == isplit):
            raise ValueError(f"No slices assigned to {split}; increase --n-slices")
    print(f"Slice edges ({args.slice_mode}): {np.array2string(edges, precision=4)}")
    print("Split per slice: " + " ".join(SPLITS[i][0] for i in split_of_slice))

    binned = [bin_event_by_slice(e, edges) for e in events]
    del events

    outfile = h5py.File(outpath, "w")
    split_groups = {split: outfile.create_group(split) for split in SPLITS}

    slicing_group = outfile.create_group("slicing")
    slicing_group.create_dataset("edges", data=edges)
    slicing_group.create_dataset("split_of_slice", data=split_of_slice)
    slicing_group.attrs["mode"] = args.slice_mode
    slicing_group.attrs["n_per_overlay"] = args.n_per_overlay
    slicing_group.attrs["n_overlays"] = args.n_overlays
    slicing_group.attrs["replace"] = args.replace
    slicing_group.attrs["source_file"] = inpath
    if args.seed is not None:
        slicing_group.attrs["seed"] = args.seed

    # Pooled (not per-slice-averaged) training statistics: per-slice phi std is only the
    # slice width, so averaging per-slice spherings would badly under-scale phi.
    nfeatures = binned[0][0].shape[1]
    train_n = 0
    train_sum = np.zeros(nfeatures)
    train_sumsq = np.zeros(nfeatures)

    nhits = {split: [] for split in SPLITS}
    nhits_by_slice = [[] for _ in range(args.n_slices)]
    islice_out = 0
    for ioverlay in range(args.n_overlays):
        chosen = rng.choice(len(binned), size=args.n_per_overlay, replace=args.replace)
        for islice in range(args.n_slices):
            split = SPLITS[split_of_slice[islice]]
            data = np.concatenate([binned[ievt][islice] for ievt in chosen])

            # Written unsphered for now; training/validation are sphered in place below
            slice_group = split_groups[split].create_group(f"slice_{islice_out}")
            slice_group.create_dataset("tau0", data=data)
            slice_group.attrs["overlay"] = ioverlay
            slice_group.attrs["phi_slice"] = islice
            slice_group.attrs["phi_range"] = edges[islice:islice + 2]
            islice_out += 1

            if split == "training":
                train_n += len(data)
                data64 = data.astype(np.float64)
                train_sum += data64.sum(axis=0)
                train_sumsq += (data64**2).sum(axis=0)
            nhits[split].append(len(data))
            nhits_by_slice[islice].append(len(data))
        print(f"Built overlay {ioverlay + 1}/{args.n_overlays}")

    mu = train_sum / train_n
    std = np.sqrt(train_sumsq / train_n - mu**2)
    sphering = Sphering(mu, std)

    sphere_group = outfile.create_group("transformation")
    sphere_group.create_dataset("mu", data=mu)
    sphere_group.create_dataset("std", data=std)
    sphere_group.attrs["log_energy"] = log_energy
    sphere_group.attrs["log_time"] = log_time

    for split in ("training", "validation"):
        for slice_group in split_groups[split].values():
            slice_group["tau0"][...] = sphering.transform(slice_group["tau0"][...])

    outfile.close()

    print("\nnhits per slice:")
    for split in SPLITS:
        n = np.array(nhits[split])
        if len(n):
            print(f"  {split:>10}: {len(n)} slices, mean={n.mean():.5g}, std={n.std():.4g}, "
                  f"range=[{n.min()}, {n.max()}]")
    slice_means = np.array([np.mean(n) for n in nhits_by_slice])
    print(f"  mean nhits by phi slice: {np.array2string(slice_means, precision=0, max_line_width=120)}")
    print(f"  spread of slice means: std/mean = {slice_means.std() / slice_means.mean():.3g}")
    print(f"Wrote {outpath}")

    return 0

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build full-BIB overlays from make_training_data.py output, chunk them by phi slice, "
                                                 "and write slices in make_training_data.py format with train/val/test assigned by phi slice")
    parser.add_argument("infile", help="Processed HDF5 file (output of make_training_data.py --cylindrical)")
    parser.add_argument("-o", "--out", default="overlay_data.hdf5", help="Path to output")
    parser.add_argument("-k", "--n-per-overlay", type=int, default=1666, help="Number of source events overlaid into one full BIB event")
    parser.add_argument("-n", "--n-overlays", type=int, default=1, help="Number of full BIB overlays to build")
    parser.add_argument("--n-slices", type=int, default=16, help="Number of phi slices per overlay")
    parser.add_argument("--slice-mode", choices=("width", "population"), default="width",
                        help="width: equal-width phi slices; population: edges at pooled phi quantiles so expected nhits per slice is equal")
    parser.add_argument("-p", "--phi-window", type=float, default=np.pi,
                        help="Slices span [-p, p]. Should match the -p used for make_training_data.py")
    parser.add_argument("-s", "--split", default="0.7,0.2,0.1",
                        help="training,validation,test fractions of phi slices; slices are interleaved around phi")
    parser.add_argument("--replace", action="store_true", help="Draw source events within an overlay with replacement (default: without)")
    parser.add_argument("--seed", type=int, default=None, help="RNG seed for reproducibility")
    print("\nFinished with exit code:", main(parser.parse_args()))
