import numpy as np
from enum import Enum


class ObservationType(Enum):
    """Observation type for trajectories.

    - FULL: Entire function is known via a callable
    - DENSE: Densely sampled observations over the domain
    - SPARSE: Sparsely sampled observations over the domain
    """
    FULL = "full"
    DENSE = "dense"
    SPARSE = "sparse"

class Trajectory:
    """Represents a single functional trajectory.

    Parameters
    ----------
    observation_type : ObservationType | str
        Observation type: 'full' | 'dense' | 'sparse'.
    domain : tuple[float, float]
        Function domain (a, b) with a < b.
    t : array_like, optional
        Observation time points (required for dense/sparse).
    y : array_like, optional
        Observed values (required for dense/sparse). Supports 1D or 2D shape.
        If 1D of length m, it is treated as (m, 1). If 2D, shape must be (m, d).
    func : callable, optional
        Callable representing the function (required for full). Signature f(t) -> y, where y is 1D array of length d.
    regular : bool, optional
        True if time points are equally spaced (only for dense/sparse). Default False.

    Attributes
    ----------
    m : int
        Number of observation points; equals 0 for full.
    t : np.ndarray | None
        Observation time points; None for full.
    y : np.ndarray | None
        Observed values as a 2D array of shape (m, d); None for full.
    func : callable | None
        Function representation for full; None for dense/sparse.
    observation_type : ObservationType
        Observation type.
    domain : tuple[float, float]
        Function domain.
    dim : int | None
        Output dimension d. For full, inferred from func if not provided.
    regular : bool
        True if time points are equally spaced (only meaningful for dense/sparse).
    """

    def __init__(self, observation_type, domain, t=None, y=None, func=None, regular=False, output_dim=None):
        # Normalize and validate observation type
        if isinstance(observation_type, str):
            observation_type = observation_type.lower()
            if observation_type not in {"full", "dense", "sparse"}:
                raise ValueError("observation_type must be one of 'full', 'dense', 'sparse'.")
            observation_type = ObservationType(observation_type)
        elif not isinstance(observation_type, ObservationType):
            raise TypeError("observation_type must be an ObservationType or str.")

        # Validate domain
        if not (isinstance(domain, (tuple, list)) and len(domain) == 2):
            raise ValueError("domain must be a tuple/list of length 2: (a, b).")
        a, b = float(domain[0]), float(domain[1])
        if not (a < b):
            raise ValueError("domain must satisfy a < b.")
        self.domain = (a, b)

        self.observation_type = observation_type
        self.regular = bool(regular)

        if observation_type == ObservationType.FULL:
            if func is None or not callable(func):
                raise ValueError("For FULL observation, a callable 'func' must be provided.")
            self.func = func
            self.t = None
            self.y = None
            self.m = 0
            # Determine output dimension if provided or infer by sampling mid-domain
            if output_dim is not None:
                if not isinstance(output_dim, int) or output_dim <= 0:
                    raise ValueError("output_dim must be a positive integer if provided.")
                self.dim = output_dim
            else:
                mid = (a + b) / 2.0
                sample = np.asarray(func(mid))
                if sample.ndim == 0:
                    sample = sample.reshape(1)
                elif sample.ndim != 1:
                    raise ValueError("func(mid) must return a 1D array-like output.")
                self.dim = int(sample.shape[0])
        else:
            # For dense/sparse, t and y are required
            if t is None or y is None:
                raise ValueError("For DENSE/SPARSE observation, both 't' and 'y' must be provided.")
            t = np.array(t, dtype=float)
            y = np.array(y, dtype=float)
            if t.ndim != 1:
                raise ValueError("t must be a 1D array of time points.")
            if y.ndim == 1:
                if y.shape[0] != t.shape[0]:
                    raise ValueError("Length of y must match length of t when y is 1D.")
                y = y.reshape(-1, 1)
            elif y.ndim == 2:
                if y.shape[0] != t.shape[0]:
                    raise ValueError("y must have shape (m, d) where m == len(t).")
            else:
                raise ValueError("y must be 1D or 2D array.")

            self.t = t
            self.y = y
            self.m = int(t.shape[0])
            self.dim = int(y.shape[1])
            self.func = None

        # Validate time points lie within domain for dense/sparse
        if self.t is not None:
            if np.any(self.t < a) or np.any(self.t > b):
                raise ValueError("All time points 't' must lie within the given domain.")

    def get_dim(self):
        """Return the output dimension d of the trajectory.

        For full trajectories, this is inferred during initialization (or via output_dim).
        For dense/sparse trajectories, this equals y.shape[1].
        """
        return self.dim

    def __repr__(self):
        ot = self.observation_type.value
        return (
            f"Trajectory(type={ot}, m={self.m}, dim={self.dim}, domain={self.domain}, "
            f"regular={self.regular})"
        )


class FunctionalData:
    """Functional dataset consisting of multiple trajectories.

    Parameters
    ----------
    trajectories : list[Trajectory]
        List of trajectories. All must share the same observation type, domain, and output dimension.
    domain : tuple[float, float], optional
        Dataset domain. If provided, must match all trajectories.

    Attributes
    ----------
    trajectories : list[Trajectory]
        Trajectory list.
    n : int
        Number of trajectories (samples).
    observation_type : ObservationType | None
        Observation type of the dataset or None if dataset is empty.
    domain : tuple[float, float] | None
        Dataset function domain or None if dataset is empty.
    dim : int | None
        Output dimension of trajectories or None if dataset is empty.
    dense : bool
        True if observation type is DENSE.
    regular : bool
        True if all trajectories are regular (only meaningful for dense/sparse).
    """

    def __init__(self, trajectories, domain=None):
        self.trajectories = trajectories or []
        self.n = len(self.trajectories)

        if self.n == 0:
            self.observation_type = None
            self.domain = tuple(domain) if domain is not None else None
            self.dim = None
            self.dense = False
            self.regular = False
            return

        # Validate and unify observation type
        types = [traj.observation_type for traj in self.trajectories]
        first_type = types[0]
        if not all(t == first_type for t in types):
            raise ValueError("All trajectories must have the same observation_type (full/dense/sparse).")
        self.observation_type = first_type

        # Validate and unify domain
        domains = [traj.domain for traj in self.trajectories]
        if domain is not None:
            a, b = float(domain[0]), float(domain[1])
            if not (a < b):
                raise ValueError("domain must satisfy a < b.")
            dataset_domain = (a, b)
            if not all(d == dataset_domain for d in domains):
                raise ValueError("All trajectories must share the same domain as the dataset domain.")
            self.domain = dataset_domain
        else:
            first_domain = domains[0]
            if not all(d == first_domain for d in domains):
                raise ValueError("All trajectories must share the same domain.")
            self.domain = first_domain

        # Validate and unify output dimension
        dims = [traj.get_dim() for traj in self.trajectories]
        first_dim = dims[0]
        if not all(d == first_dim for d in dims):
            raise ValueError("All trajectories must share the same output dimension 'dim'.")
        self.dim = first_dim

        # Compatibility attributes
        self.dense = self.observation_type == ObservationType.DENSE
        self.regular = all(traj.regular for traj in self.trajectories)

    def __repr__(self):
        ot = self.observation_type.value if self.observation_type is not None else None
        return (
            f"FunctionalData(n={self.n}, type={ot}, dim={self.dim}, domain={self.domain}, "
            f"dense={self.dense}, regular={self.regular})"
        )
