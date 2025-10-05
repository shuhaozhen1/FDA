"""
Demo: simulate sparse functional dataset using KL-based 5D random process
per the provided paper settings, plot a few trajectories, and print dataset
properties for verification.
"""

import os
import numpy as np
import matplotlib.pyplot as plt

from FDA.generators import generate_sparse_dataset


def main():
    n = 50  # sample size
    m = 5   # baseline observations per trajectory
    snr = 3.0  # between 2 and 5
    domain = (0.0, 1.0)

    fd, trajectories, sigma = generate_sparse_dataset(n=n, m=m, snr=snr, domain=domain, K=20)

    print("Dataset:", fd)
    print("Noise std (sigma):", sigma)
    print("First trajectory dim:", trajectories[0].get_dim())
    print("Observation type:", trajectories[0].observation_type)
    print("Domain:", trajectories[0].domain)

    # Plot first 3 trajectories' first two dimensions
    out_dir = os.path.join(os.path.dirname(__file__), "outputs")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "simulation_samples.png")

    plt.figure(figsize=(9, 6))
    colors = ["tab:blue", "tab:orange", "tab:green"]
    for idx in range(3):
        traj = trajectories[idx]
        plt.scatter(traj.t, traj.y[:, 0], s=25, marker="o", alpha=0.8, c=colors[idx], label=f"traj{idx+1} y[0]")
        plt.scatter(traj.t, traj.y[:, 1], s=25, marker="x", alpha=0.8, c=colors[idx], label=f"traj{idx+1} y[1]")

    plt.title("Sparse simulated trajectories (dims 1 and 2)")
    plt.xlabel("t")
    plt.ylabel("y components")
    plt.legend(ncol=2)
    plt.grid(True, linestyle="--", alpha=0.4)
    plt.tight_layout()
    plt.savefig(out_path)
    print(f"Saved plot to: {out_path}")


if __name__ == "__main__":
    main()