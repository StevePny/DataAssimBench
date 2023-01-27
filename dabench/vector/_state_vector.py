"""Class for storing state vector(s) and accompanying info"""

import numpy as np
import jax.numpy as jnp
from dabench.vector import _vector


class StateVector(_vector._Vector):
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
                 x0=None,
                 **kwargs):
        self._values = None
        self._xi = None
        self.original_dim = original_dim

        super().__init__(system_dim=system_dim,
                         time_dim=time_dim,
                         delta_t=delta_t,
                         store_as_jax=store_as_jax)

        self.x0 = x0

    def __str__(self):
        return f'Current State = {self.xi}, Timesteps = {self.time_dim}'
                
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
    def values_gridded(self):
        if self._values is None:
            return None
        else:
            return self._to_original_dim()

    @property
    def x0(self):
        if self._x0 is None and self.values is not None:
            self.x0 = self.values[0]
        return self._x0

    @x0.setter
    def x0(self, x0_vals):
        if x0_vals is None:
            self._x0 = None
        else:
            if self.store_as_jax:
                self._x0 = jnp.asarray(x0_vals)
            else:
                self._x0 = np.asarray(x0_vals)

    @x0.deleter
    def x0(self):
        del self._x0

    @property
    def x0_gridded(self):
        if self._x0 is None:
            return None
        else:
            return self._x0.reshape(self.original_dim)

    @property
    def xi(self):
        if self._xi is None:
            if self.values is not None:
                self.xi = self.values[-1]
            elif self.x0 is not None:
                self.xi = self.x0
        return self._xi

    @xi.setter
    def xi(self, xi_vals):
        if xi_vals is None:
            self._xi = None
        else:
            if self.store_as_jax:
                self._xi = jnp.asarray(xi_vals)
            else:
                self._xi = np.asarray(xi_vals)

    @xi.deleter
    def xi(self):
        del self._xi

    @property
    def xi_gridded(self):
        if self._xi is None:
            return None
        else:
            return self._xi.reshape(self.original_dim)

    def _to_original_dim(self):
        """Converts 1D representation of system back to original dimensions.

        Returns:
            Multidimensional array with shape:
            (time_dim, original_dim[0], ..., original_dim[n])
        """
        return self.values.reshape((self.time_dim,) + self.original_dim)
