"""
Demo: create two sparse trajectories, plot them, and validate FunctionalData.

This script:
- Creates two sparse Trajectory instances over the same domain with 2D outputs
- Plots both trajectories and saves to PNG
- Builds FunctionalData from them and prints dataset properties for verification
"""

import os
import numpy as np
import matplotlib.pyplot as plt

from FDA.functional_data import Trajectory, FunctionalData


def create_sparse_trajectories():
    domain = (0.0, 5.0)

    # Trajectory 1: y = [sin(t), cos(t)] sampled sparsely
    t1 = np.array([0.2, 0.8, 1.7, 3.3, 4.7])
    y1 = np.stack([np.sin(t1), np.cos(t1)], axis=1)  # shape (m1, 2)
    traj1 = Trajectory("sparse", domain, t=t1, y=y1)

    # Trajectory 2: y = [sin(2t), cos(2t)] sampled sparsely
    t2 = np.array([0.1, 1.0, 2.4, 3.8, 4.9])
    y2 = np.stack([np.sin(2 * t2), np.cos(2 * t2)], axis=1)  # shape (m2, 2)
    traj2 = Trajectory("sparse", domain, t=t2, y=y2)

    return domain, traj1, traj2


def plot_sparse(traj1: Trajectory, traj2: Trajectory, out_path: str):
    plt.figure(figsize=(8, 5))

    # Plot trajectory 1 components
    plt.scatter(traj1.t, traj1.y[:, 0], c="tab:blue", label="traj1 y[0]=sin(t)")
    plt.scatter(traj1.t, traj1.y[:, 1], c="tab:orange", label="traj1 y[1]=cos(t)")

    # Plot trajectory 2 components
    plt.scatter(traj2.t, traj2.y[:, 0], c="tab:green", label="traj2 y[0]=sin(2t)")
    plt.scatter(traj2.t, traj2.y[:, 1], c="tab:red", label="traj2 y[1]=cos(2t)")

    plt.title("Sparse trajectories (2D outputs)")
    plt.xlabel("t")
    plt.ylabel("y components")
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.4)
    plt.tight_layout()
    plt.savefig(out_path)
    print(f"Saved plot to: {out_path}")


def verify_functional_data(domain, traj1, traj2):
    fd = FunctionalData([traj1, traj2])
    print("FunctionalData representation:", fd)
    print("n:", fd.n)
    print("type:", fd.observation_type)
    print("domain:", fd.domain)
    print("dim:", fd.dim)
    print("dense:", fd.dense)
    print("regular:", fd.regular)

    # Additional checks
    assert fd.observation_type.value == "sparse"
    assert fd.domain == domain
    assert fd.dim == 2
    assert fd.dense is False


def main():
    domain, traj1, traj2 = create_sparse_trajectories()
    out_dir = os.path.join(os.path.dirname(__file__), "outputs")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "sparse_trajectories.png")
    plot_sparse(traj1, traj2, out_path)
    verify_functional_data(domain, traj1, traj2)


if __name__ == "__main__":
    main()