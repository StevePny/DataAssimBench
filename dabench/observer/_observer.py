"""Base class for Observer object

Input is  generated data, returns ObsVector with values, times, coords, etc
"""

import numpy as np
import jax.numpy as jnp

from dabench.vector import ObsVector

rng = np.random.default_rng(45)


class Observer():
    """Base class for Observer objects

    Attributes:
        data_obj (dabench.data.Data): Data generator/loader object from which
            to gather observations.
        locations (ndarray): Locations to gather observations from. If not
            specified, will be randomly generated according to loc_density.
            Default is None.
        loc_density (float): Fraction of locations to gather observations from,
            must be value between 0 and 1. Default is 1.
        times (ndarray): Times to gather observations from. If not specified,
            randomly generate according to time_density. Default is None.
        time_density (float): Fraction of times to gather observations from,
            must be value between 0 and 1. Default is 1.
        error_bias (float): Mean of normal distribution of observation errors.
            Default is 0.
        error_sd (float): Standard deviation of observation errors. Default is
            0.
    """

    def __init__(self,
                 data_obj,
                 locations=None,
                 loc_density=1.,
                 times=None,
                 time_density=1.,
                 error_bias=0.,
                 error_sd=0.
                 ):
        self.data_obj = data_obj
        self.locations = locations
        self.loc_density = loc_density
        self.times = times
        self.time_density = time_density
        self.error_bias = error_bias
        self.error_sd = error_sd

    def observe(self):
        """Generate observations.

        Returns:
            ObsVector containing observation values, times, locations, and
                errors
        """

        if self.data_obj.values is None:
            self.data_obj.generate()

        loc_vector = rng.binomial(1, p=self.loc_density,
                                  size=self.data_obj.system_dim).astype('bool')
        time_vector = rng.binomial(1, p=self.time_density,
                                   size=self.data_obj.time_dim).astype('bool')

        errors_vector = rng.normal(loc=self.error_bias, scale=self.error_sd,
                                   size=(time_vector.sum(),
                                         loc_vector.sum()))
                                        

        values_vector = (self.data_obj.values[time_vector][:, loc_vector]
                         + errors_vector)

        return ObsVector(values=values_vector,
                         times=np.repeat(time_vector, loc_vector.shape[0]),
                         coords=np.repeat(loc_vector, time_vector.shape[0]),
                         errors=errors_vector,
                         error_dist='normal'
                         )
