import argparse

import h5py
import numpy as np

from BIBgen.preprocessing import Sphering
from BIBgen.analysis import ComparisonAnalyzer

def main(args):
    mcpath = args.mc_file
    outpath = args.out
    gen_input = {}
    for entry in args.gen_files:
        entry_split = entry.split(",")
        assert len(entry_split) == 2, "Each gen-files entry should be a tuple of path and name"
        assert entry_split[0].endswith(".hdf5")
        gen_input[entry_split[1]] = (entry_split[0], entry_split[1].lower())
      
    assert mcpath.endswith(".hdf5")

    with h5py.File(mcpath, "r") as mcfile:
        mu = np.array(mcfile["transformation/mu"])
        std = np.array(mcfile["transformation/std"])

        mcdata = {event_id : np.array(mcfile["test/" + event_id + "/tau0"]) for event_id in mcfile["test"].keys()}

    # print("mc nhits =", len(mcdata))

    gendata = {}
    for name in gen_input:
        with h5py.File(gen_input[name][0], "r") as genfile:
            gendata[name] = {event_id : np.array(genfile[event_id]) for event_id in genfile.keys()}

    analyzer = ComparisonAnalyzer(
        energy_range = (-0.0005, 0.005),
        phi_range = (-1.0, 1.0),
        eta_range = (-1.3, 1.3),
        s_range = (1800, 2250),
        z_range = (-2800, 2800),
        output_dir = outpath
    )
    analyzer.load_from_dict("MC", mcdata, is_sphered=False)
    for name in gendata:
        analyzer.load_from_dict(name, gendata[name], sphering=Sphering(mu, std))

    for genname in gen_input:
        analyzer.plot_kinematics_1d("MC", genname, prefix=gen_input[genname][1], normalized=False, log_scale=False)

    analyzer.plot_s_eta_2d("MC", "mc")
    for genname in gen_input:
        analyzer.plot_s_eta_2d(genname, prefix=gen_input[genname][1])

    for event_id in mcdata:
        analyzer.plot_clustering(event_id, prefix=event_id, reference_key="MC", use_energy=True)

    return 0

if __name__ == "__main__":
    # uv run plot_comparison.py ../data/raw_cyl_phipi4_large.hdf5 generation/like_v9.hdf5,Deepsets generation/like_v10.hdf5,MLP
    parser = argparse.ArgumentParser()
    parser.add_argument("mc_file")
    parser.add_argument("gen_files", nargs="+")
    parser.add_argument("-o", "--out", default="plots")
    print("\nFinished with exit code:", main(parser.parse_args()))