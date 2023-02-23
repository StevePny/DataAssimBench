"""Class for storing an observation vector and accompanying info"""

import copy

import numpy as np
import jax.numpy as jnp

from dabench.vector import _vector


class ObsVector(_vector._Vector):
    """Class for storing observations

    Attributes:
        num_obs (int): Number of observations.
        obs_dims (array): Number of values stored in each observation.
        values (array): array of observation values. If each observation can
            store more than one value (e.g. wind and temperature at some
            location), values is 2D with first dimension num_obs and variable
            second dimension of lengths obs_dims.
        coords (ndarray): n-dimensional array of locations associated with 
            each observation. For example, 2D if only x and y coordinates, 3D
            if x, y, and z, etc.
        errors (array): 1d array of errors associated with each observation
        error_dist (str): String describing error distribution (e.g. Gaussian)
        times (array): 1d array of times associated with each observation
        store_as_jax (bool): Store values as jax array instead of numpy array.
            Default is False (store as numpy).
        """
    def __init__(self,
                 num_obs=None,
                 obs_dims=None,
                 error_dist=None,
                 values=None,
                 coords=None,
                 errors=None,
                 times=None,
                 store_as_jax=False,
                 **kwargs):

        self.obs_dims = obs_dims
        self.num_obs = num_obs
        self.error_dist = error_dist

        super().__init__(times=times,
                         store_as_jax=store_as_jax,
                         values=values,
                         **kwargs)

        self.coords = coords
        self.errors = errors

    @property
    def coords(self):
        return self._coords

    @coords.setter
    def coords(self, vals):
        if vals is None:
            self._coords = None
        else:
            if self.store_as_jax:
                self._coords = jnp.asarray(vals)
            else:
                self._coords = np.asarray(vals)

    @coords.deleter
    def coords(self):
        del self._coords

    @property
    def errors(self):
        return self._errors

    @errors.setter
    def errors(self, vals):
        if vals is None:
            self._errors = None
        else:
            if self.store_as_jax:
                self._errors = jnp.asarray(vals)
            else:
                self._errors = np.asarray(vals)

    @errors.deleter
    def errors(self):
        del self._errors

    def filter_times(self, start, end, inclusive=True):
        """Filter observations to within a time range, returns copy of object

        Args:
            start (datetime or float): Start of time range. Type must match
                type of times (e.g. datetime if times are datetimes, float if
                times are floats). Default is None, which includes all
                values up to "end". If neither start nor end are specified,
                time_filter does nothing.
            end (datetime or float): Start of time range. See "start" for
                more information. Default is None, which includes all values
                after "start".
            inclusive (bool): If True, includes times that are equal to start
                or end. If False, excludes times equal to start/end Default is
                True.

        Returns:
            Copy of object with filtered values.
        """
        new_vec = copy.deepcopy(self)
        if start is not None:
            if inclusive:
                filtered_idx = new_vec.times >= start
            else:
                filtered_idx = new_vec.times > start
            new_vec.times = new_vec.times[filtered_idx]
            new_vec.values = new_vec.values[filtered_idx]
            if new_vec.errors is not None:
                new_vec.errors = new_vec.errors[filtered_idx]
            if new_vec.coords is not None:
                new_vec.coords = new_vec.coords[filtered_idx]

        if end is not None:
            if inclusive:
                filtered_idx = new_vec.times <= end
            else:
                filtered_idx = new_vec.times < end
            new_vec.times = new_vec.times[filtered_idx]
            new_vec.values = new_vec.values[filtered_idx]
            if new_vec.errors is not None:
                new_vec.errors = new_vec.errors[filtered_idx]
            if new_vec.coords is not None:
                new_vec.coords = new_vec.coords[filtered_idx]

        new_vec.num_obs = new_vec.values.shape[0]
        new_vec.obs_dims = self.obs_dims[filtered_idx]

        return new_vec

