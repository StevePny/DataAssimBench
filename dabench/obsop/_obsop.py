"""Base class for ObsOp object

Input is a StateVector from a Model, returns ObsVector
"""

import warnings

import numpy as np

from dabench.vector import StateVector, ObsVector


class ObsOp():
    """Base class for ObsOp objects

    Attributes:
        H (ndarray): Observation operator matrix that maps from model to
            observation. Can be manually specified or generated according to
            location_indices or random_location_count. If not provided,
            will be calculated when ObsOp.observe() is run. Default is identity
            matrix with shape (state_vec.system_dim, state_vec.system_dim).
        location_indices (ndarray): Indices to gather observations from. If
            location_indices is provided, will create H. Default is None.
        random_location_count (int): Number of locations to randomly select
            for gathering observations from. If random_location_count is
            provided, will generate location_indices and H accordingly. Default
            is None.
        random_seed (int): Random seed for sampling locations. Default is 99.

    """
    def __init__(self,
                 H=None,
                 location_indices=None,
                 random_location_count=None,
                 random_seed=99
                 ):
        self.location_indices = location_indices
        self.random_location_count = random_location_count
        self.random_seed = random_seed
        self.H = H
        self._rng = np.random.default_rng(self.random_seed)

    def _generate_indices(self, state_vec):
        return self._rng.choice(
                state_vec.system_dim,
                size=self.random_location_count,
                replace=False,
                shuffle=False
                )

    def _get_H(self, state_vec):
        if self.random_location_count is not None:
            if self.location_indices is not None:
                warnings.warn(
                    'Both location_indices and random_location_count were '
                    'provided. Proceeding using location_indices and '
                    'ignoring random_location_count.'
                    )
            else:
                self.location_indices = self._generate_indices(state_vec)

        if self.location_indices is not None:
            return np.take(
                    np.identity(state_vec.system_dim),
                    self.location_indices,
                    axis=0)
        else:
            return np.identity(state_vec.system_dim)

    def observe(self, state_vec):
        """Generate observations according to ObsOp attributes

        Args:
            state_vec (dabench.vector.StateVector): StateVector input.

        Returns:
            Observation vector (dabench.vector.ObsVector).
        """
        # Initialize H
        if self.H is None:
            self.H = self._get_H(state_vec)

        out_values = self.H * state_vec.values

        return ObsVector(
                values=out_values,
                times=state_vec.times)
