import argparse

import h5py
import numpy as np

from BIBgen.preprocessing import diffuse

def main(args):
    inpath = args.raw_data
    outpath = args.out
    schedule_path = args.noise_schedule
    assert inpath.endswith(".hdf5")
    assert schedule_path.endswith(".csv")

    schedule = np.loadtxt(schedule_path)
    alpha_bar = np.prod(1 - schedule)
    print("alpha_bar =", alpha_bar)

    infile = h5py.File(inpath, "r")
    outfile = h5py.File(outpath, "w")

    sphere_group = outfile.create_group("transformation")
    sphere_group.create_dataset("mu", data=infile["transformation/mu"])
    sphere_group.create_dataset("std", data=infile["transformation/std"])

    for dataset in ("training", "validation"):
        ds_group = outfile.create_group(dataset)

        for event_id in infile[dataset].keys():
            tau0 = infile[dataset + "/" + event_id + "/tau0"]

            ds_group.create_dataset(event_id, data=diffuse(tau0, schedule))
            print("Processed {} for {}".format(event_id, dataset))

    infile.close()
    outfile.close()
    return 0

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run diffusion on preprocessed data")
    parser.add_argument("raw_data", help="Undiffused raw data produced by make_training_data.py")
    parser.add_argument("noise_schedule")
    parser.add_argument("-o", "--out", default="diffused.hdf5", help="Path to output path")
    print("\nFinished with exit code:", main(parser.parse_args()))