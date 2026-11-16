import re
import argparse

import numpy as np
import matplotlib.pyplot as plt

TAG_RE = re.compile(r"history_(.+)\.csv$")

def main(args):
    for path in args.histories:
        epoch, val_loss = np.loadtxt(path, delimiter=",", skiprows=1, unpack=True)
        match = TAG_RE.search(path)
        label = match.group(1) if match else path
        plt.plot(epoch, val_loss, label=label)

    plt.xlabel("Epoch")
    plt.ylabel("Validation loss")
    if args.log_y:
        plt.yscale("log")
    plt.legend()
    plt.title("Validation loss history comparison", fontweight="bold")
    plt.tight_layout()

    if args.out:
        plt.savefig(args.out)
        print(f"Wrote comparison plot to {args.out}")
    else:
        plt.show()
    return 0

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Plot and compare validation loss histories (history_<tag>.csv files from train.py)")
    parser.add_argument("histories", nargs="+", help="One or more history_<tag>.csv paths")
    parser.add_argument("--log-y", action="store_true", help="Plot validation loss on a log scale")
    parser.add_argument("-o", "--out", default=None, help="Path to save the comparison plot (default: show interactively)")
    print("\nFinished with exit code:", main(parser.parse_args()))
