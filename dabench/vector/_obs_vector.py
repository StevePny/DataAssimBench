"""Class for storing an observation vector and accompanying info"""

import numpy as np
import jax.numpy as jnp
from dabench.vector import _vector


class ObsVector(_vector._Vector):
    """Class for storing state(s) of a system

    Attributes:
        system_dim (int): system dimension
        time_dim (int): total time steps
        original_dim (tuple): dimensions in original space, e.g. could be 3x3
            for a 2d system with system_dim = 9. 
        delta_t (float): the timestep of the data (assumed uniform)
        values (ndarray): 2d array of data (time_dim, system_dim),
        store_as_jax (bool): Store values as jax array instead of numpy array.
            Default is False (store as numpy).
        """

    def __init__(self,
                 system_dim=None,
                 time_dim=None,
                 original_dim=None,
                 delta_t=None,
                 store_as_jax=False,
                 error_dist=None,
                 times=None,
                 **kwargs):
        self._values = None
        self._locations = None
        self._errors = None
        self.times = times
        self.error_dist = error_dist
        self.original_dim = original_dim

        super().__init__(system_dim=system_dim,
                         time_dim=time_dim,
                         delta_t=delta_t,
                         store_as_jax=store_as_jax)

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

    @values.deleter
    def values(self):
        del self._values

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
