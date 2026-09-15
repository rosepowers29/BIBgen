"""
Scans a generated .hdf5 event-by-event (not aggregated) and reports the max absolute
whitened feature magnitude per event, sorted descending -- isolates which specific
event(s) are behind an aggregate-level blowup (see inspect_generated_energy.py) so a
targeted, single-event -v 2 rerun of generate_like.py can trace the exact tau it happens at.

Pure numpy/h5py -- no torch, no GPU needed, safe to run directly on an access point.
"""
import argparse

import h5py
import numpy as np

def main(args):
    with h5py.File(args.gen_file, "r") as f:
        stats = []
        for event_id in f.keys():
            data = np.array(f[event_id])
            stats.append((event_id, data.shape[0], np.abs(data).max()))

    stats.sort(key=lambda row: row[2], reverse=True)

    print(f"{'event_id':<12} {'n_hits':>8} {'max |whitened value|':>22}")
    for event_id, n_hits, max_val in stats[:args.top_n]:
        print(f"{event_id:<12} {n_hits:>8} {max_val:>22.6g}")

    return 0

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Find which event(s) in a generated .hdf5 have diverged")
    parser.add_argument("gen_file", help="Generated .hdf5 (e.g. cosine_predvar_fulltest_like.hdf5)")
    parser.add_argument("-n", "--top-n", type=int, default=15, help="Number of worst events to print (default: 15)")
    print("\nFinished with exit code:", main(parser.parse_args()))
