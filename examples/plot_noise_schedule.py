import argparse

import numpy as np
import matplotlib.pyplot as plt

def main(args):
    schedules = [np.loadtxt(p) for p in args.schedules]
    labels = args.labels or args.schedules

    for label, schedule in zip(labels, schedules):
        alpha_bar = np.cumprod(1 - schedule)
        print(
            f"{label}: T={len(schedule)}, beta_min={schedule.min():.3e}, "
            f"beta_max={schedule.max():.3e}, alpha_bar_T={alpha_bar[-1]:.3e}"
        )

    if args.out:
        fig, (ax_beta, ax_abar) = plt.subplots(2, 1, figsize=(6, 6), constrained_layout=True)
        for label, schedule in zip(labels, schedules):
            tau = np.arange(1, len(schedule) + 1)
            alpha_bar = np.cumprod(1 - schedule)
            ax_beta.plot(tau, schedule, label=label)
            ax_abar.plot(tau, alpha_bar, label=label)
        ax_beta.set_ylabel(r"$\beta_\tau$")
        ax_abar.set_ylabel(r"$\bar\alpha_\tau$")
        ax_abar.set_yscale("log")
        ax_abar.set_xlabel(r"$\tau$")
        ax_beta.legend()
        fig.suptitle("Noise schedule comparison", fontweight="bold")
        plt.savefig(args.out)
        print(f"Wrote comparison plot to {args.out}")
    return 0

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Print/plot beta(tau) and alpha_bar(tau) for one or more schedules")
    parser.add_argument("schedules", nargs="+", help="One or more noise schedule CSV paths")
    parser.add_argument("--labels", nargs="+", default=None, help="Labels for each schedule (defaults to the paths)")
    parser.add_argument("-o", "--out", default=None, help="Optional path to save a comparison PNG")
    print("\nFinished with exit code:", main(parser.parse_args()))
