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

        test_events = list(mcfile["test"].keys())
        mcdata = np.array(mcfile["test/" + test_events[0] + "/tau0"])

    print("mc nhits =", len(mcdata))

    with h5py.File(genpath, "r") as genfile:
        print(genfile)
        gendata = {event_id : np.array(genfile[event_id]) for event_id in genfile.keys()}

    for event_id in gendata:
        analyzer = BIBgenHistogramAnalyzer(outpath)
        mc_vars = analyzer.load_from_model_output(mcdata, is_sphered=False)
        gen_vars = analyzer.load_from_model_output(gendata[event_id], sphering=Sphering(mu, std))

        analyzer.plot_overlay_comparison(mc_vars, gen_vars, prefix=event_id)
        analyzer.plot_eta_phi_2d(mc_vars, prefix="mc")
        analyzer.plot_eta_phi_2d(gen_vars, prefix="gen")
        analyzer.plot_delta_r_clustering(mc_vars, prefix="mc")
        analyzer.plot_delta_r_clustering(gen_vars, prefix="gen")
        break

    # noise_vars = analyzer.load_from_model_output(np.random.normal(size=(len(mcdata), 4)), sphering=Sphering(mu, std))
    # analyzer.plot_overlay_comparison(mc_vars, noise_vars, prefix="noise")

    return 0

if __name__ == "__main__":
    # uv run plot_comparison.py ../data/raw_cyl_phipi4_medium.hdf5 generation/demo_v4_n5077.hdf5
    parser = argparse.ArgumentParser()
    parser.add_argument("mc_file")
    parser.add_argument("gen_file")
    parser.add_argument("-o", "--out", default="plots")
    print("\nFinished with exit code:", main(parser.parse_args()))