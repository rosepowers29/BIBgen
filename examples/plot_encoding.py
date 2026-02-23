import argparse

import numpy as np
import matplotlib.pyplot as plt
import h5py
import torch

def main(args):
    model_path = args.model_path
    raw_path = args.raw_data_path
    outpath = args.out
    assert model_path.endswith(".pth") and raw_path.endswith(".hdf5") and outpath.endswith(".png")

    weights = torch.load(model_path, weights_only=True, map_location=torch.device('cpu'))

    omega_phi = weights["pos1_encoding.frequency_table"]
    omega_s = weights["pos2_encoding.frequency_table"]
    omega_z = weights["pos3_encoding.frequency_table"]

    with h5py.File(raw_path, "r") as fin:
        _, phi_std, s_std, z_std = np.array(fin["transformation/std"])
    
    T_phi = phi_std * 2*np.pi / omega_phi
    T_s = s_std * 2*np.pi / omega_s
    T_z = z_std * 2*np.pi / omega_z

    fig, axes = plt.subplots(3, figsize=(6, 3), constrained_layout=True)
    fig.suptitle('Periods of learned Fourier encoding', fontweight='bold')

    alpha = 0.3
    axes[0].scatter(T_phi, np.zeros_like(T_phi), alpha=alpha)
    axes[0].set_xlabel(r"$T_{\phi}$ [rad]")
    axes[1].scatter(T_s, np.zeros_like(T_s), alpha=alpha)
    axes[1].set_xlabel(r"$T_{s}$ [mm]")
    axes[2].scatter(T_z, np.zeros_like(T_z), alpha=alpha)
    axes[2].set_xlabel(r"$T_{z}$ [mm]")

    for i in range(3):
        axes[i].set_yticks([])
        # axes[i].set_xscale("log")

    plt.savefig(outpath)
    return 0

if __name__ == "__main__":
    # uv run plot_encoding.py training/denoiser_v7.pth ../data/raw_cyl_phipi4_large.hdf5
    parser = argparse.ArgumentParser()
    parser.add_argument("model_path")
    parser.add_argument("raw_data_path")
    parser.add_argument("-o", "--out", default="periods.png")
    print("\nFinished with exit code:", main(parser.parse_args()))