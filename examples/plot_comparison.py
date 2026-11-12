import argparse
import csv
import os
import re

import h5py
import numpy as np
from scipy.stats import wasserstein_distance

from BIBgen.preprocessing import Sphering
from BIBgen.analysis import ComparisonAnalyzer

WASSERSTEIN_VARIABLES = ("energy", "phi", "eta", "s", "z")

def infer_tag(genpath):
    stem = os.path.splitext(os.path.basename(genpath))[0]
    return re.sub(r"_(like|gen)$", "", stem)

def write_wasserstein_distances(analyzer, reference_name, tags, outdir):
    """
    Writes per-variable Wasserstein (earth-mover's) distance between the MC
    and each generated distribution to <outdir>/wasserstein_distances.csv.
    Tag is included as a column so per-run CSVs can be concatenated later
    for a cross-experiment comparison.
    """
    mc_vars = analyzer.aggr_data[reference_name]
    distances = {
        name : {k: wasserstein_distance(mc_vars[k], analyzer.aggr_data[name][k]) for k in WASSERSTEIN_VARIABLES}
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

        mcdata = {event_id : np.array(mcfile["test/" + event_id + "/tau0"]) for event_id in mcfile["test"].keys()}

    if args.log_energy == "auto":
        log_energy = stored_log_energy
    else:
        log_energy = (args.log_energy == "yes")
        if log_energy != stored_log_energy:
            print(f"Warning: --log-energy={args.log_energy} overrides the log_energy={stored_log_energy} "
                  f"flag stored in {mcpath}")

    # Generated files from generate_like.py carry no attributes and are sphered with the MC file's
    # energy parameterisation. from_cache.py writes physical units and says so via file attributes.
    gendata = {}
    gen_sphered = {}
    gen_log_energy = {}
    for name in gen_input:
        with h5py.File(gen_input[name][0], "r") as genfile:
            gendata[name] = {event_id : np.array(genfile[event_id]) for event_id in genfile.keys()}
            gen_sphered[name] = bool(genfile.attrs.get("sphered", True))
            gen_log_energy[name] = bool(genfile.attrs.get("log_energy", log_energy))

    print(f"Writing plots to {outpath}")

    analyzer = ComparisonAnalyzer(
        energy_range = (-0.0005, 0.005),
        phi_range = (-1.0, 1.0),
        eta_range = (-1.3, 1.3),
        s_range = (1800, 2250),
        z_range = (-2800, 2800),
        output_dir = outpath
    )
    analyzer.load_from_dict("MC", mcdata, is_sphered=False, exponentiate_energy=log_energy)
    for name in gendata:
        analyzer.load_from_dict(name, gendata[name], is_sphered=gen_sphered[name], sphering=Sphering(mu, std),
                                exponentiate_energy=gen_log_energy[name])

    for genname in gen_input:
        analyzer.plot_kinematics_1d("MC", genname, prefix=gen_input[genname][1], normalized=False, log_scale=False)

    write_wasserstein_distances(analyzer, "MC", tags, outpath)

    analyzer.plot_s_eta_2d("MC", "mc")
    for genname in gen_input:
        analyzer.plot_s_eta_2d(genname, prefix=gen_input[genname][1])

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
    print("\nFinished with exit code:", main(parser.parse_args()))
