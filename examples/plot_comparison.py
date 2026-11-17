import argparse
import csv
import os
import re

import h5py
import numpy as np
from scipy.stats import wasserstein_distance

from BIBgen.preprocessing import Sphering
from BIBgen.analysis import ComparisonAnalyzer

WASSERSTEIN_VARIABLES = ("energy", "phi", "eta", "s", "z", "time")
WASSERSTEIN_2D_PAIRS = (("eta", "phi"), ("s", "z"), ("eta", "s"), ("eta", "z"),
        ("s", "phi"), ("z", "phi"))

def none_or_int(value):
    return None if value.lower() == "none" else int(value)

def none_or_float(value):
    return None if value.lower() == "none" else float(value)

def infer_tag(genpath):
    stem = os.path.splitext(os.path.basename(genpath))[0]
    return re.sub(r"_like$", "", stem)

def write_wasserstein_distances(analyzer, reference_name, tags, outdir):
    """
    Writes per-variable Wasserstein (earth-mover's) distance between the MC
    and each generated distribution to <outdir>/wasserstein_distances.csv.
    Tag is included as a column so per-run CSVs can be concatenated later
    for a cross-experiment comparison.
    """
    mc_vars = analyzer.aggr_data[reference_name]
    variables = [v for v in WASSERSTEIN_VARIABLES if v in mc_vars]
    distances = {
        name : {k: wasserstein_distance(mc_vars[k], analyzer.aggr_data[name][k]) for k in variables}
        for name in tags
    }

    csv_path = os.path.join(outdir, "wasserstein_distances.csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["tag", "variable", "wasserstein_distance"])
        for name in distances:
            for k, v in distances[name].items():
                writer.writerow([tags[name], k, v])

    print(f"Wasserstein distances written to {csv_path}:")
    for name in distances:
        print(f"  {name} (tag: {tags[name]}):")
        for k, v in distances[name].items():
            print(f"    {k}: {v:.6g}")

    return distances

def write_wasserstein_distances_2d(analyzer, reference_name, tags, outdir,
                                    pairs=WASSERSTEIN_2D_PAIRS, max_samples=200, normalize=True, seed=0):
    """
    Writes per-pair N-D Wasserstein distance (e.g. the joint eta-phi or s-z
    distribution) between the MC and each generated distribution to
    <outdir>/wasserstein_distances_2d.csv. Mirrors write_wasserstein_distances,
    but kept in a separate file/column layout ("variables" holds e.g. "eta-phi")
    since analyze_ablation.py's pivot on wasserstein_distances.csv expects one
    row per single variable name.

    With normalize=True (default), each variable is min-max scaled against its
    configured range before the distance is computed, so mixed-unit pairs (e.g.
    eta alongside z) are comparable instead of being dominated by whichever
    variable has the larger raw magnitude -- see
    ComparisonAnalyzer.compute_wasserstein_nd for the full units discussion.
    """
    mc_vars = analyzer.aggr_data[reference_name]
    pairs = [p for p in pairs if all(v in mc_vars for v in p)]
    distances = {
        name: analyzer.compute_wasserstein_nd_pairs(reference_name, name, pairs, max_samples=max_samples, normalize=normalize, seed=seed)
        for name in tags
    }

    csv_path = os.path.join(outdir, "wasserstein_distances_2d.csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["tag", "variables", "wasserstein_distance", "normalized"])
        for name in distances:
            for pair, v in distances[name].items():
                writer.writerow([tags[name], "-".join(pair), v, normalize])

    units_note = "normalized to configured ranges, dimensionless" if normalize else "raw units"
    print(f"2D Wasserstein distances ({units_note}) written to {csv_path}:")
    for name in distances:
        print(f"  {name} (tag: {tags[name]}):")
        for pair, v in distances[name].items():
            print(f"    {'-'.join(pair)}: {v:.6g}")

    return distances

def drop_diverged_events(events, name, max_sphered):
    """
    Removes generated events that diverged during sampling: any non-finite value, or (if
    max_sphered is not None) any |x| > max_sphered in sphered units. Such events would
    otherwise overflow in the exp/expm1 inverse transform and poison the plots and
    Wasserstein distances. Prints how many were dropped so the divergence rate can be
    tracked across checkpoints.
    """
    kept, nonfinite, out_of_range = {}, [], []
    for event_id, x in events.items():
        if not np.isfinite(x).all():
            nonfinite.append(event_id)
        elif max_sphered is not None and x.size > 0 and np.abs(x).max() > max_sphered:
            out_of_range.append(event_id)
        else:
            kept[event_id] = x

    n_dropped = len(nonfinite) + len(out_of_range)
    if n_dropped > 0:
        print(f"Warning: dropped {n_dropped} / {len(events)} ({100 * n_dropped / len(events):.1f}%) "
              f"diverged events from '{name}': {len(nonfinite)} non-finite"
              + (f", {len(out_of_range)} with |x| > {max_sphered} (sphered)" if max_sphered is not None else ""))
        print(f"  non-finite: {' '.join(nonfinite) if nonfinite else '-'}")
        print(f"  out-of-range: {' '.join(out_of_range) if out_of_range else '-'}")
    else:
        print(f"No diverged events in '{name}' ({len(events)} events)")
    return kept

def main(args):
    mcpath = args.mc_file
    gen_input = {}
    for entry in args.gen_files:
        entry_split = entry.split(",")
        assert len(entry_split) == 2, "Each gen-files entry should be a tuple of path and name"
        assert entry_split[0].endswith(".hdf5")
        gen_input[entry_split[1]] = (entry_split[0], entry_split[1].lower())

    assert mcpath.endswith(".hdf5")

    # Experiment tag per generated dataset: what the Wasserstein rows are keyed by, so that
    # analyze_ablation.py can join them against history_<tag>.csv. --tag overrides the tag
    # inferred from the file name only when there is a single generated file to attach it to.
    if args.tag is not None and len(gen_input) == 1:
        tags = {name : args.tag for name in gen_input}
    else:
        tags = {name : infer_tag(gen_input[name][0]) for name in gen_input}
    outpath = args.out or os.path.join("plots", tags[next(iter(gen_input))])

    with h5py.File(mcpath, "r") as mcfile:
        mu = np.array(mcfile["transformation/mu"])
        std = np.array(mcfile["transformation/std"])
        stored_log_energy = bool(mcfile["transformation"].attrs.get("log_energy", False))
        stored_log_time = bool(mcfile["transformation"].attrs.get("log_time", False))

        mcdata = {event_id : np.array(mcfile["test/" + event_id + "/tau0"]) for event_id in mcfile["test"].keys()}

    if args.log_energy == "auto":
        log_energy = stored_log_energy
    else:
        log_energy = (args.log_energy == "yes")
        if log_energy != stored_log_energy:
            print(f"Warning: --log-energy={args.log_energy} overrides the log_energy={stored_log_energy} "
                  f"flag stored in {mcpath}")

    if args.log_time == "auto":
        log_time = stored_log_time
    else:
        log_time = (args.log_time == "yes")
        if log_time != stored_log_time:
            print(f"Warning: --log-time={args.log_time} overrides the log_time={stored_log_time} "
                  f"flag stored in {mcpath}")

    gendata = {}
    for name in gen_input:
        with h5py.File(gen_input[name][0], "r") as genfile:
            gendata[name] = {event_id : np.array(genfile[event_id]) for event_id in genfile.keys()}
        gendata[name] = drop_diverged_events(gendata[name], name, args.max_sphered)

    print(f"Writing plots to {outpath}")

    analyzer = ComparisonAnalyzer(
        energy_range = (-0.0005, 0.005),
        phi_range = (-1.0, 1.0),
        eta_range = (-1.3, 1.3),
        s_range = (1800, 2250),
        z_range = (-2800, 2800),
        time_range = (0, 15000),
        output_dir = outpath
    )
    analyzer.load_from_dict("MC", mcdata, is_sphered=False, exponentiate_energy=log_energy, exponentiate_time=log_time)
    for name in gendata:
        analyzer.load_from_dict(name, gendata[name], sphering=Sphering(mu, std), exponentiate_energy=log_energy, exponentiate_time=log_time)

    for genname in gen_input:
        analyzer.plot_kinematics_1d("MC", genname, prefix=gen_input[genname][1], normalized=False, log_scale=False)

    write_wasserstein_distances(analyzer, "MC", tags, outpath)
    write_wasserstein_distances_2d(analyzer, "MC", tags, outpath, max_samples=args.wasserstein_2d_samples)

    analyzer.plot_s_eta_2d("MC", "mc")
    for genname in gen_input:
        analyzer.plot_s_eta_2d(genname, prefix=gen_input[genname][1])
        analyzer.plot_eta_phi_residual_2d("MC", genname, prefix=gen_input[genname][1])

    return 0

if __name__ == "__main__":
    # uv run plot_comparison.py /ospool/uc-shared/project/futurecolliders/rosep8/raw_cyl_phipi4_large_logE.hdf5 generation/v13_like.hdf5,v13 -t v13
    parser = argparse.ArgumentParser()
    parser.add_argument("mc_file")
    parser.add_argument("gen_files", nargs="+", help="One or more <path.hdf5>,<name> entries; name is the legend label")
    parser.add_argument("-o", "--out", default=None, help="Output directory for plots (default: plots/<tag>)")
    parser.add_argument("-t", "--tag", default=None,
        help="Experiment tag used to pick the output subdirectory, and the Wasserstein tag when a "
             "single generated file is given (default: inferred from the generated file name)")
    parser.add_argument("-l", "--log-energy", choices=["auto", "yes", "no"], default="auto",
        help="Whether the energy feature is ln(E). 'auto' reads the flag stored in the MC file.")
    parser.add_argument("--log-time", choices=["auto", "yes", "no"], default="auto",
        help="Whether the time feature is ln(1+t). 'auto' reads the flag stored in the MC file.")
    parser.add_argument("--max-sphered", type=none_or_float, default=10.0,
        help="Drop generated events with any |value| above this in sphered units (diverged "
             "samples); non-finite events are always dropped. 'none' drops only non-finite events.")
    parser.add_argument("--wasserstein-2d-samples", type=none_or_int, default=200,
        help="Per-dataset subsample cap for the 2D (eta-phi, s-z) Wasserstein distances, or "
             "'none' to disable subsampling -- wasserstein_distance_nd solves an exact transport "
             "LP that scales poorly with sample count, so keep this small.")
    print("\nFinished with exit code:", main(parser.parse_args()))
