import numpy as np

class Trajectory:
    """
    Represents a single trajectory of functional data.

    Parameters
    ----------
    t : array_like
        Observation time points.
    y : array_like
        Observed values corresponding to each time point.
    dense : bool, optional
        Whether this trajectory is considered dense. Default is False.
    regular : bool, optional
        Whether time points are regularly spaced. Default is False.

    Attributes
    ----------
    m : int
        Number of observation points in this trajectory.
    t : np.ndarray
        Observation time points as a numpy array.
    y : np.ndarray
        Observed values as a numpy array.
    dense : bool
        True if this trajectory is considered dense.
    regular : bool
        True if the time points are regularly spaced.
    """

    def __init__(self, t, y, dense=False, regular=False):
        t = np.array(t, dtype=float)
        y = np.array(y, dtype=float)

        if len(t) != len(y):
            raise ValueError("Time points array (t) and observations array (y) must have the same length.")

        self.t = t
        self.y = y
        self.m = len(t)
        self.dense = dense
        self.regular = regular

    def __repr__(self):
        return f"Trajectory(m={self.m}, dense={self.dense}, regular={self.regular})"


class FunctionalData:
    """
    A collection of multiple Trajectory objects.

    Parameters
    ----------
    trajectories : list
        A list of Trajectory objects.

    Attributes
    ----------
    trajectories : list
        The list of Trajectory objects.
    n : int
        Total number of trajectories in this functional dataset.
    dense : bool
        True if all trajectories are dense; otherwise False.
    regular : bool
        True if all trajectories are regular; otherwise False.
    """

    def __init__(self, trajectories):
        self.trajectories = trajectories
        self.n = len(trajectories) if trajectories else 0

        if self.n > 0:
            self.dense = all(traj.dense for traj in trajectories)
            self.regular = all(traj.regular for traj in trajectories)
        else:
            self.dense = False
            self.regular = False

    def __repr__(self):
        return (f"FunctionalData(n={self.n}, dense={self.dense}, "
                f"regular={self.regular})")
