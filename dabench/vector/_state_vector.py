"""Class for storing state vector(s) and accompanying info"""

import numpy as np
import jax.numpy as jnp
from dabench.vector import _vector


class StateVector(_vector._Vector):
    """Class for storing state(s) of a system. 

    Notes:
        - Can store a single "state vector", meaning the state of the system
            at a single timestep.
        - Can also store multiple state vectors (in other words, a trajectory).
        - Shape of StateVector().values will always be (time_dim, system_dim).

    Attributes:
        system_dim (int): system dimension
        time_dim (int): total time steps
        original_dim (tuple): dimensions in original space, e.g. could be 3x3
            for a 2d system with system_dim = 9. Default is None.
        delta_t (float): the timestep of the data (assumed uniform)
        store_as_jax (bool): Store values as jax array instead of numpy array.
            Default is False (store as numpy).
        values (ndarray): 2d array of data (time_dim, system_dim). Default is
            None.
        times (array): 1D array of times associated with each state. If not
            provided, will be set to np.arange(time_dim). Default is None.
        xi (ndarray): Most recent state of the system. Default is None.
        """

    def __init__(self,
                 system_dim=None,
                 time_dim=None,
                 original_dim=None,
                 delta_t=None,
                 store_as_jax=False,
                 values=None,
                 times=None,
                 **kwargs):
        self._xi = None
        self.system_dim = system_dim
        self.delta_t = delta_t
        if original_dim is None:
            self.original_dim = system_dim
        else:
            self.original_dim = original_dim

        super().__init__(time_dim=time_dim,
                         store_as_jax=store_as_jax,
                         values=values,
                         times=times,
                         **kwargs)

    def __str__(self):
        return f'Current State = {self.xi}, Timesteps = {self.time_dim}'

    @property
    def values_gridded(self):
        if self._values is None:
            return None
        else:
            return self._to_original_dim()

    @property
    def xi(self):
        if self._xi is None and self.values is not None:
            self.xi = self.values[-1]
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
