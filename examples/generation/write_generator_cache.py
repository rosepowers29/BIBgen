from __future__ import annotations

import argparse

from BIBgen.generation import Generator


def main(args):
    training_file = args.training_file
    outpath = args.out
    assert outpath.endswith(".npz")

    generator = Generator()
    generator.calculate_cache(training_file)
    generator.save_cache(outpath)
    print("Wrote cache of {} event sizes to {}".format(len(generator.sorted_nhits), outpath))

    return 0

# uv run write_generator_cache.py ../../data/raw_cyl_phipi4_large.hdf5 -o generator_cache.npz
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Caches the event size distribution and sphering of training data for use by from_cache.py")
    parser.add_argument("training_file", help="Path to training data (preferably undiffused)")
    parser.add_argument("-o", "--out", default="generator_cache.npz", help="Path to output cache")
    print("\nFinished with exit code:", main(parser.parse_args()))
