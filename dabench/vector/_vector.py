"""Base class for vector objects"""

import numpy as np
import jax.numpy as jnp

class _Vector():
    """Base class for vector objects

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
                 **kwargs):

        self.system_dim = system_dim
        self.time_dim = time_dim
        self.original_dim = original_dim
        self.delta_t = delta_t
        self.store_as_jax = store_as_jax

