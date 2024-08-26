"""Class for storing an observation vector and accompanying info"""

import copy
import warnings

import numpy as np
import jax.numpy as jnp

from dabench.vector import _vector


class ObsVector(_vector._Vector):
    """Class for storing observations

    Attributes:
        num_obs (int): Number of observations.
        obs_dims (array): Number of values stored in each observation. Must
            match dims of values array. If not specified, will be calculated
            using values. Default is None.
        values (array): array of observation values. If each observation can
            store more than one value (e.g. wind and temperature at some
            location), values is 2D with first dimension num_obs and variable
            second dimension of lengths obs_dims.
        coords (ndarray): n-dimensional array of locations associated with
            each observation. For example, 2D if only x and y coordinates, 3D
            if x, y, and z, etc.
        time_indices (ndarray): Array of indices in data generator object
            time_dim from which observations were made. Default is None.
        location_indices (ndarray): Array of indices in data generator object
            values from which observations were made. Default is None.
        errors (array): 1d array of errors associated with each observation
        error_dist (str): String describing error distribution (e.g. Gaussian)
        error_sd (float): If applicable, standard deviation of Gaussian dist
            from which errors were sampled. Default is None.
        error_bias (float): If applicable,mean of Gaussian dist from which
            errors were sampled. Default is None.
        times (array): 1d array of times associated with each observation
        store_as_jax (bool): Store values as jax array instead of numpy array.
            Default is False (store as numpy).
        stationary_observers (bool): If True, samples are from same indices at
            each time step. If False, observations can be from different
            indices, including irregular numbers of observations and different
            time steps.
        """
    def __init__(self,
                 num_obs=None,
                 obs_dims=None,
                 error_dist=None,
                 values=None,
                 coords=None,
                 time_indices=None,
                 location_indices=None,
                 errors=None,
                 error_sd=None,
                 error_bias=None,
                 times=None,
                 store_as_jax=False,
                 stationary_observers=True,
                 **kwargs):

        self.num_obs = num_obs
        self.error_dist = error_dist
        self.error_sd = error_sd
        self.error_bias = error_bias
        self.time_indices = time_indices
        self.location_indices = location_indices
        self.stationary_observers = stationary_observers

        super().__init__(times=times,
                         store_as_jax=store_as_jax,
                         **kwargs)

        self.values = values

        # Calculate/check obs_dims
        if self.values is not None:
            self._calc_obs_dims()
            # Check user provided obs_dims
            if obs_dims is not None:
                if not np.array_equal(obs_dims, self.obs_dims):
                    warnings.warn('obs_dims {} does not match dimensions of '
                                  ' values {}.\n Proceeding with obs_dims '
                                  'calculated based on values..'.format(
                                      obs_dims, self.obs_dims))
        else:
            self.obs_dims = obs_dims

        self.coords = coords
        self.errors = errors

    def __getitem__(self, subscript):
        if self.values is None:
            raise AttributeError('Object does not contain any data values.\n'
                                 'Run .generate() or .load() and try again')

        if isinstance(subscript, slice):
            new_copy = copy.deepcopy(self)
            new_copy.values = new_copy.values[
                    subscript.start:subscript.stop:subscript.step]
            new_copy.times = new_copy.times[
                    subscript.start:subscript.stop:subscript.step]
            new_copy.location_indices = new_copy.location_indices[subscript]
            if new_copy.errors is not None:
                new_copy.errors = new_copy.errors[subscript]
            if new_copy.coords is not None:
                new_copy.coords = new_copy.coords[subscript]
            return new_copy
        else:
            new_copy = copy.deepcopy(self)
            new_copy.values = new_copy.values[subscript]
            new_copy.times = new_copy.times[subscript]
            new_copy.location_indices = new_copy.location_indices[subscript]
            if new_copy.errors is not None:
                new_copy.errors = new_copy.errors[subscript]
            if new_copy.coords is not None:
                new_copy.coords = new_copy.coords[subscript]
            return new_copy

    @property
    def values(self):
        return self._values

    @values.setter
    def values(self, vals):
        if vals is None:
            self._values = None
        else:
            if self.store_as_jax:
                self._values = jnp.asarray(vals)
            else:
                self._values = np.asarray(vals)
            self.num_obs = self._values.shape[0]
            self._calc_obs_dims()

    @values.deleter
    def values(self):
        del self._values

    def _calc_obs_dims(self):
        """Private helper method for calculating obs_dims"""
        if self.values.dtype is np.dtype('O'):
            self.obs_dims = np.array([v.shape[0] for
                                      v in self.values])
        elif len(self.values.shape) == 1:
            self.obs_dims = np.repeat(1, self.values.shape[0])
        else:
            self.obs_dims = np.repeat(
                    self.values.shape[1],
                    self.values.shape[0])

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
