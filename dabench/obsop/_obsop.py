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
            location_indices. If not provided, will be calculated when
            ObsOp.observe() is run. Default is identity matrix with shape
            (state_vec.system_dim, state_vec.system_dim).
        location_indices (ndarray): Indices to gather observations from. If
            location_indices is provided, will create H. Default is None.
    """
    def __init__(self,
                 H=None,
                 location_indices=None,
                 ):
        self.location_indices = location_indices
        self.H = H

    def _get_H(self, state_vec):
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
