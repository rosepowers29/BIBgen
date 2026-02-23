import argparse

import h5py
import numpy as np

from BIBgen.preprocessing import Sphering
from BIBgen.analysis import BIBgenHistogramAnalyzer

def main(args):
    mcpath = args.mc_file
    genpath = args.gen_file
    outpath = args.out
    assert mcpath.endswith(".hdf5") and genpath.endswith(".hdf5")

    with h5py.File(mcpath, "r") as mcfile:
        mu = np.array(mcfile["transformation/mu"])
        std = np.array(mcfile["transformation/std"])

        # test_events = list(mcfile["test"].keys())
        mcdata = {event_id : np.array(mcfile["test/" + event_id + "/tau0"]) for event_id in mcfile["test"].keys()}

    # print("mc nhits =", len(mcdata))

    with h5py.File(genpath, "r") as genfile:
        gendata = {event_id : np.array(genfile[event_id]) for event_id in genfile.keys()}

    aggr_gendata = np.concatenate(list(gendata.values()))
    aggr_mcdata = np.concatenate(list(mcdata.values()))

    analyzer = BIBgenHistogramAnalyzer(
        energy_range = (-0.0005, 0.005),
        phi_range = (-1.0, 1.0),
        eta_range = (-1.3, 1.3),
        s_range = (1800, 2250),
        z_range = (-2800, 2800),
        output_dir = outpath
    )
    mc_vars = analyzer.load_from_model_output(aggr_mcdata, is_sphered=False)
    gen_vars = analyzer.load_from_model_output(aggr_gendata, sphering=Sphering(mu, std))

    analyzer.plot_overlay_comparison(mc_vars, gen_vars, prefix="aggr_log", normalized=False)
    analyzer.plot_overlay_comparison(mc_vars, gen_vars, prefix="aggr", normalized=False, log_scale=False)
    analyzer.plot_eta_phi_2d(mc_vars, prefix="mc", bins=50)
    analyzer.plot_eta_phi_2d(gen_vars, prefix="gen", bins=50)
    analyzer.plot_s_eta_2d(mc_vars, prefix="mc", bins=50)
    analyzer.plot_s_eta_2d(gen_vars, prefix="gen", bins=50)
    analyzer.plot_delta_r_clustering(mc_vars, prefix="mc")
    analyzer.plot_delta_r_clustering(gen_vars, prefix="gen")

    # noise_vars = analyzer.load_from_model_output(np.random.normal(size=(len(mcdata), 4)), sphering=Sphering(mu, std))
    # analyzer.plot_overlay_comparison(mc_vars, noise_vars, prefix="noise")

    return 0

if __name__ == "__main__":
    # uv run plot_comparison.py ../data/raw_cyl_phipi4_large.hdf5 generation/like_v7.hdf5
    parser = argparse.ArgumentParser()
    parser.add_argument("mc_file")
    parser.add_argument("gen_file")
    parser.add_argument("-o", "--out", default="plots")
    print("\nFinished with exit code:", main(parser.parse_args()))