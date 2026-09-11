import argparse
import os

import numpy as np

from BIBgen.preprocessing import quadratic_beta_schedule, cosine_beta_schedule

def main(args):
    outpath = args.out
    assert outpath.endswith(".csv")
    if os.path.exists(outpath) and not args.force:
        raise SystemExit(f"{outpath} already exists; pass --force to overwrite.")

    if args.type == "quadratic":
        schedule = quadratic_beta_schedule(args.n_timesteps, scale=args.scale)
    else:
        schedule = cosine_beta_schedule(
            args.n_timesteps,
            s=args.s,
            target_alpha_bar_T=args.target_alpha_bar_t,
            beta_clip=args.beta_clip,
        )

    alpha_bar_T = np.prod(1 - schedule)
    print(
        f"Generated {args.type} schedule: T={args.n_timesteps}, "
        f"beta_min={schedule.min():.3e}, beta_max={schedule.max():.3e}, "
        f"alpha_bar_T={alpha_bar_T:.3e}"
    )
    np.savetxt(outpath, schedule)
    print(f"Wrote schedule to {outpath}")
    return 0

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate a beta(tau) noise schedule CSV")
    parser.add_argument("type", choices=["quadratic", "cosine"])
    parser.add_argument("out", help="Output CSV path (np.loadtxt-compatible, one beta per line)")
    parser.add_argument("-T", "--n-timesteps", type=int, default=100)
    parser.add_argument("--scale", type=float, default=3e-5, help="[quadratic] beta(tau) = scale * tau^2")
    parser.add_argument("--s", type=float, default=0.008, help="[cosine] offset (Nichol & Dhariwal)")
    parser.add_argument("--target-alpha-bar-t", type=float, default=1e-5, help="[cosine] target alpha_bar at tau=T")
    parser.add_argument("--beta-clip", type=float, default=0.999, help="[cosine] defensive max beta")
    parser.add_argument("-f", "--force", action="store_true", help="Overwrite out if it already exists")
    print("\nFinished with exit code:", main(parser.parse_args()))
