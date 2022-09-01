"""Uniform Random data generator for basic testing.

    Typical usage example:

    rdata = Random()
    rdata.generate()
"""
import jax.random as jrandom
from data import data


class Random(data.Data):
    """ Class to set up Random model data as simple test case. NOT for testing.

    Attributes:
        system_dim (int): system dimension
        time_dim (int): total time steps
        original_dim (tuple): dimensions in original space, e.g. could be 3x3
            for a 2d system with system_dim = 9. Defaults to (system_dim),
            i.e. 1d.
        random_seed (int): random seed, defaults to 37
        values (:obj: `ndarray`): 2d array of data (time_dim, system_dim)
    """

    def __init__(self,
                 system_dim=3,
                 time_dim=1,
                 original_dim=None,
                 random_seed=37,
                 values=None,
                 **kwargs):
        """Initializes the Random data object"""

        super().__init__(system_dim=system_dim,
                         time_dim=time_dim,
                         original_dim=original_dim,
                         random_seed=random_seed,
                         values=values)

    def generate(self):
        """Generates random trajectory for system

        Values are 2d array of data of dimension (time_dim, system_dim)
        """

        key = jrandom.PRNGKey(self.random_seed)
        self.values = jrandom.uniform(key, (self.time_dim, self.system_dim))
