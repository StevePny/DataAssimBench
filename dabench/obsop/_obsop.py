"""Base class for ObsOp object

Input is a StateVector from a Model, returns ObsVector
"""

import warnings
import inspect

import numpy as np
import jax.numpy as jnp

from dabench import vector


class ObsOp():
    """Base class for ObsOp objects

    Attributes:
        h (function): Operator that takes state_vector and existing obs_vector
            with locations of observations as inputs. Outputs new obs_vector.
            If not provided, will default to simple indexing of the
            state_vector. Default is None.
        H (ndarray): Linearization of operator h. Not currently supported.
    """
    def __init__(self,
                 h=None,
                 H=None):
        if h is None:
            self.h = self._index_state_vec
        else:
            self.h = h
        self.H = H

    def _index_state_vec(self, state_vec, obs_vector=None):
        if obs_vector is not None and obs_vector.location_indices is not None:
            location_indices = obs_vector.location_indices
        else:
            location_indices = np.arange(state_vec.system_dim)

        out_vals = [state_vec.values[i][np.array(location_indices)]
                    for i in range(state_vec.time_dim)]

        return vector.ObsVector(
                values=out_vals,
                times=state_vec.times,
                time_dim=state_vec.time_dim,
                location_indices=location_indices,
                store_as_jax=state_vec.store_as_jax
                )

