import os
import sys
import numpy as np
import pytest

# add project root to sys.path for imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from FDA.functional_data import Trajectory, FunctionalData, ObservationType


def test_full_trajectory_and_dataset():
    func = lambda t: np.array([np.sin(t), np.cos(t)])
    domain = (0.0, 2.0)
    traj = Trajectory("full", domain, func=func)
    assert traj.observation_type == ObservationType.FULL
    assert traj.func is not None
    assert traj.t is None and traj.y is None
    assert traj.domain == domain
    assert traj.get_dim() == 2

    fd = FunctionalData([traj])
    assert fd.n == 1
    assert fd.observation_type == ObservationType.FULL
    assert fd.domain == domain
    assert fd.dim == 2
    assert fd.dense is False


def test_dense_trajectory_and_dataset():
    domain = (0.0, 1.0)
    t = np.linspace(domain[0], domain[1], 50)
    y = np.stack([np.cos(t), np.sin(t)], axis=1)  # shape (50, 2)
    traj = Trajectory("dense", domain, t=t, y=y, regular=True)
    assert traj.observation_type == ObservationType.DENSE
    assert traj.m == len(t)
    assert traj.regular is True
    assert traj.get_dim() == 2

    fd = FunctionalData([traj, traj])
    assert fd.n == 2
    assert fd.observation_type == ObservationType.DENSE
    assert fd.domain == domain
    assert fd.dim == 2
    assert fd.dense is True
    assert fd.regular is True


def test_sparse_trajectory_and_dataset_domain_check():
    domain = (0.0, 1.0)
    t = np.array([0.1, 0.3, 0.9])
    y = np.stack([np.ones_like(t), 0.5 * np.ones_like(t)], axis=1)  # (3,2)
    traj = Trajectory("sparse", domain, t=t, y=y)
    assert traj.observation_type == ObservationType.SPARSE
    assert traj.domain == domain
    assert traj.get_dim() == 2

    fd = FunctionalData([traj])
    assert fd.observation_type == ObservationType.SPARSE
    assert fd.domain == domain
    assert fd.dim == 2

    # domain mismatch in dataset initialization should raise
    with pytest.raises(ValueError):
        FunctionalData([traj], domain=(0.0, 2.0))


def test_mixed_types_raise():
    domain = (0.0, 1.0)
    t = np.linspace(0.0, 1.0, 10)
    y = np.stack([np.sin(t), np.cos(t)], axis=1)
    dense_traj = Trajectory("dense", domain, t=t, y=y)
    full_traj = Trajectory("full", domain, func=lambda x: np.array([x, x**2]))
    with pytest.raises(ValueError):
        FunctionalData([dense_traj, full_traj])


def test_t_outside_domain_raises():
    domain = (0.0, 1.0)
    t = np.array([-0.1, 0.5])
    y = np.array([0.0, 1.0])
    with pytest.raises(ValueError):
        Trajectory("dense", domain, t=t, y=y)


def test_dim_mismatch_raises():
    domain = (0.0, 1.0)
    t = np.linspace(0.0, 1.0, 5)
    y1 = np.stack([t, t**2], axis=1)  # dim=2
    y2 = np.stack([t, t, t], axis=1)  # dim=3
    traj1 = Trajectory("dense", domain, t=t, y=y1)
    traj2 = Trajectory("dense", domain, t=t, y=y2)
    with pytest.raises(ValueError):
        FunctionalData([traj1, traj2])