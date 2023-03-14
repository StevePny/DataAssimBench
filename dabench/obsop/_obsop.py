"""Base class for ObsOp object

Input is a StateVector from a Model, returns ObsVector
"""

import warnings

import numpy as np
import jax.numpy as jnp

from dabench import vector


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

        if self.location_indices is None:
            # Default: all locations
            self.location_indices = np.arange(state_vec.system_dim)

        return np.take(
                np.identity(state_vec.system_dim),
                self.location_indices,
                axis=0)

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

        if state_vec.store_as_jax:
            self.H = jnp.array(self.H)

        # Apply observation operator
        out_vals = [self.H @ state_vec.values[i]
                    for i in range(state_vec.time_dim)]

        return vector.ObsVector(
                values=out_vals,
                times=state_vec.times,
                time_dim=state_vec.time_dim,
                location_indices=self.location_indices,
                store_as_jax=state_vec.store_as_jax
                )