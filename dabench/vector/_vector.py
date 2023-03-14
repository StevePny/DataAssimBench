"""Base class for vector objects"""

import numpy as np

import jax.numpy as jnp


class _Vector():
    """Base class for vector objects

    Attributes:
        system_dim (int): system dimension
        time_dim (int): total time steps
        delta_t (float): the timestep of the data (assumed uniform)
        values (ndarray): array of data values
        store_as_jax (bool): Store values as jax array instead of numpy array.
            Default is False (store as numpy).
        """

    def __init__(self,
                 time_dim=None,
                 times=None,
                 store_as_jax=False,
                 values=None):

        self.store_as_jax = store_as_jax
        self.values = values
        if times is None and self.values is not None:
            self.times = np.arange(self.values.shape[0])
        else:
            self.times = times
        if time_dim is None and self.times is not None:
            self.time_dim = self.times.shape[0]
        else:
            self.time_dim = time_dim

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
    def times(self):
        return self._times

    @times.setter
    def times(self, vals):
        if vals is None:
            self._times = None
        else:
            if self.store_as_jax:
                self._times = jnp.asarray(vals)
            else:
                self._times = np.asarray(vals)

    @times.deleter
    def times(self):
        del self._times
