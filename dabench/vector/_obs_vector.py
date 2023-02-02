"""Class for storing an observation vector and accompanying info"""

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

        self._locations = None
        self._errors = None

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
