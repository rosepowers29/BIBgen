import argparse
import os
import re

import h5py
import numpy as np

from BIBgen.preprocessing import Sphering
from BIBgen.analysis import BIBgenHistogramAnalyzer

from scipy.stats import wasserstein_distance
import csv

def write_wasserstein_distances(mc_vars, gen_vars, outdir, tag):
    """
    Writes per-variable Wasserstein (earth-mover's) distance between the MC
    and generated distributions to <outdir>/wasserstein_distances.csv.
    Tag is included as a column so per-run CSVs can be concatenated later
    for a cross-experiment comparison.
    """
    variables = ("energy", "phi", "eta", "s", "z")
    distances = {k: wasserstein_distance(mc_vars[k], gen_vars[k]) for k in variables}

    csv_path = os.path.join(outdir, "wasserstein_distances.csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["tag", "variable", "wasserstein_distance"])
        for k, v in distances.items():
            writer.writerow([tag, k, v])

    print(f"Wasserstein distances written to {csv_path}:")
    for k, v in distances.items():
        print(f"  {k}: {v:.6g}")

    return distances

def infer_tag(genpath):
    stem = os.path.splitext(os.path.basename(genpath))[0]
    return re.sub(r"_like$", "", stem)

def main(args):
    mcpath = args.mc_file
    genpath = args.gen_file
    assert mcpath.endswith(".hdf5") and genpath.endswith(".hdf5")

    tag = args.tag or infer_tag(genpath)
    outpath = args.out or os.path.join("plots", tag)

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

    with h5py.File(genpath, "r") as genfile:
        gendata = {event_id : np.array(genfile[event_id]) for event_id in genfile.keys()}

    aggr_gendata = np.concatenate(list(gendata.values()))
    aggr_mcdata = np.concatenate(list(mcdata.values()))

    print(f"Writing plots to {outpath}")

    analyzer = BIBgenHistogramAnalyzer(
        energy_range = (-0.0005, 0.005),
        phi_range = (-1.0, 1.0),
        eta_range = (-1.3, 1.3),
        s_range = (1800, 2250),
        z_range = (-2800, 2800),
        output_dir = outpath
    )
    mc_vars = analyzer.load_from_model_output(aggr_mcdata, is_sphered=False, exponentiate_energy=log_energy)
    gen_vars = analyzer.load_from_model_output(aggr_gendata, sphering=Sphering(mu, std), exponentiate_energy=log_energy)

    analyzer.plot_overlay_comparison(mc_vars, gen_vars, prefix="aggr_log", normalized=False)
    analyzer.plot_overlay_comparison(mc_vars, gen_vars, prefix="aggr", normalized=False, log_scale=False)
    
    write_wasserstein_distances(mc_vars, gen_vars, outpath, tag)

    analyzer.plot_eta_phi_2d(mc_vars, prefix="mc", bins=50)
    analyzer.plot_eta_phi_2d(gen_vars, prefix="gen", bins=50)
    analyzer.plot_s_eta_2d(mc_vars, prefix="mc", bins=50)
    analyzer.plot_s_eta_2d(gen_vars, prefix="gen", bins=50)
    analyzer.plot_delta_r_clustering(mc_vars, prefix="mc")
    analyzer.plot_delta_r_clustering(gen_vars, prefix="gen")

    return 0

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("mc_file")
    parser.add_argument("gen_file")
    parser.add_argument("-o", "--out", default=None, help="Output directory for plots (default: plots/<tag>)")
    parser.add_argument("-t", "--tag", default=None, help="Experiment tag used to pick the output subdirectory")
    parser.add_argument("-l", "--log-energy", choices=["auto", "yes", "no"], default="auto",
        help="Whether the energy feature is ln(E). 'auto' reads the flag stored in the MC file.")
    print("\nFinished with exit code:", main(parser.parse_args()))
