"""Class for storing an observation vector and accompanying info"""

import copy

import numpy as np
import jax.numpy as jnp

from dabench.vector import _vector


class ObsVector(_vector._Vector):
    """Class for storing observations

    Attributes:
        obs_dim (int): Number of observations
        values (array): 1d array of observations
        locations (ndarray): n-dimensional array of locations associated with 
            each observation. For example, 2D if only x and y coordinates, 3D
            if x, y, and z, etc.
        errors (array): 1d array of errors associated with each observation
        error_dist (str): String describing error distribution (e.g. Gaussian)
        times (array): 1d array of times associated with each observation
        store_as_jax (bool): Store values as jax array instead of numpy array.
            Default is False (store as numpy).
        """
    def __init__(self,
                 obs_dim=None,
                 error_dist=None,
                 values=None,
                 locations=None,
                 errors=None,
                 times=None,
                 store_as_jax=False,
                 **kwargs):

        self.obs_dim = obs_dim
        self.error_dist = error_dist
        self.locations = locations
        self.errors = errors

        super().__init__(times=times,
                         store_as_jax=store_as_jax,
                         values=values,
                         **kwargs)

    @property
    def locations(self):
        return self._locations

    @locations.setter
    def locations(self, vals):
        if vals is None:
            self._locations = None
        else:
            if self.store_as_jax:
                self._locations = jnp.asarray(vals)
            else:
                self._locations = np.asarray(vals)

    @locations.deleter
    def locations(self):
        del self._locations

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

        if end is not None:
            if inclusive:
                filtered_idx = new_vec.times <= end
            else:
                filtered_idx = new_vec.times < end
            new_vec.times = new_vec.times[filtered_idx]
            new_vec.values = new_vec.values[filtered_idx]

        return new_vec

