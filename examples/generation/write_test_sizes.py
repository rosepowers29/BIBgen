import argparse

import h5py
import numpy as np

def main(args):
    inpath = args.raw_file
    outpath = args.out
    assert inpath.endswith(".hdf5") and outpath.endswith(".csv")

    outarr = []
    with h5py.File(inpath, "r") as fin:
        for event_id in fin["test"].keys():
            outarr.append([event_id, len(fin["test/" + event_id + "/tau0"])])
    
    np.savetxt(outpath, outarr, delimiter=",", fmt='%s')

    return 0

if __name__ == "__main__":
    # uv run write_test_sizes.py ../../data/raw_cyl_phipi4_medium.hdf5 -o test_sizes_medium.csv
    # uv run write_test_sizes.py ../../data/raw_cyl_phipi4_large.hdf5 -o test_sizes_large.csv
    parser = argparse.ArgumentParser()
    parser.add_argument("raw_file")
    parser.add_argument("-o", "--out", default="test_sizes.csv")
    print("\nFinished with exit code:", main(parser.parse_args()))