"""Base class for Observer object

Input is  generated data, returns ObsVector with values, times, coords, etc
"""

import numpy as np

from dabench.vector import ObsVector

rng = np.random.default_rng(45)


class Observer():
    """Base class for Observer objects

    Attributes:
        data_obj (dabench.data.Data): Data generator/loader object from which
            to gather observations.
        location_indices (ndarray): Indices to gather observations
            from. If 1D array provided, assumed to be for flattened system_dim.
            If >1D, must have same dimensionality as data generator's
            original_dim (e.g. (x, y, z)). If stationary_observers=False,
            expects leading time dimension. If not specified, will be randomly
            generated according to location_density. Default is None.
        location_density (float or tuple): Fraction(s) of locations to gather
            observations from, must be values between 0 and 1. If one value is
            provided, samples from flattened system_dim. If tuple/list-like of
            values, length must match dimensionality of data generator's
            original_dim and each value is used to sample from its respective
            dimension.
        stationary_observers (bool): If True,
            samples from same indices at each time step. If False, randomly
            generates/expects new observation indices at each timestep.
            If False:
                If using location_density, indices are randomly generated.
                If using location_indices, expects indices to either be 2D
                (time_dim, system_dim) or >2D (time_dim, original_dim).
                Default is True .
        time_indices (ndarray): Indices of times to gather observations from.
            If not specified, randomly generate according to time_density.
            Default is None.
        time_density (float): Fraction of times to gather observations from,
            must be value between 0 and 1. Default is 1.
        error_bias (float): Mean of normal distribution of observation errors.
            Default is 0.
        error_sd (float): Standard deviation of observation errors. Default is
            0.
        error_positive_only (bool): Clip errors to be positive only. Default is
            False.
    """

    def __init__(self,
                 data_obj,
                 location_indices=None,
                 location_density=1.,
                 stationary_observer=True,
                 time_indices=None,
                 time_density=1.,
                 error_bias=0.,
                 error_sd=0.,
                 error_positive_only=False
                 ):
        self.data_obj = data_obj
        self.location_indices = location_indices
        self.location_density = location_density
        self.stationary_observer = stationary_observer
        self.time_indices = time_indices
        self.time_density = time_density
        self.error_bias = error_bias
        self.error_sd = error_sd
        self.error_positive_only = error_positive_only

    def observe(self):
        """Generate observations.

        Returns:
            ObsVector containing observation values, times, locations, and
                errors
        """

        if self.data_obj.values is None:
            raise ValueError('Data have not been generated/loaded. Run:\n'
                             'self.data_obj.generate() to create data for '
                             'observer')

        # Generate times if they aren't specifieid
        if self.time_indices is None:
            self.time_indices = np.where(
                    rng.binomial(1, p=self.time_density,
                                 size=self.data_obj.time_dim
                                 ).astype('bool')
                    )[0]
        self.time_dim = self.time_indices.shape[0]

        # Generate locations if they aren't specified
        if self.stationary_observer:
            if self.location_indices is None:
                self.location_indices = np.where(
                    rng.binomial(1, p=self.location_density,
                                 size=self.data_obj.system_dim
                                 ).astype('bool')
                    )[0]
            # Check that location_indices are in correct dimensions
            elif (len(self.location_indices.shape)
                    not in [1, len(self.data_obj.original_dim)]):
                raise ValueError('location_indices must be 1D or match\n'
                                 'self.data_obj.original_dim')
            self.location_dim = np.repeat(self.location_indices.shape,
                                          self.time_dim)

            if len(self.location_indices.shape) == 1:
                errors_vec_size = (self.time_dim,) + (self.location_dim[0],)
            else:
                errors_vec_size = ((self.time_dim,) +
                                   tuple(self.location_dim[0]))
            errors_vector = rng.normal(loc=self.error_bias,
                                       scale=self.error_sd,
                                       size=errors_vec_size)

            # Clip errors to positive only
            if self.error_positive_only:
                errors_vector[errors_vector < 0.] = 0.

            values_vector = (
                self.data_obj.values[self.time_indices][
                    :, self.location_indices]
                + errors_vector)

            # Coords is same across time_dim
            coords = np.array([self.location_indices] * self.time_dim)

        # If NON-stationary observer
        else:
            if self.location_indices is None:
                self.location_indices = np.array([
                        np.where(
                            rng.binomial(1, p=self.location_density,
                                         size=self.data_obj.system_dim
                                         ).astype('bool'))[0]
                        for i in range(self.time_indices.shape[0])
                        ], dtype=object)
            elif (len(self.location_indices.shape)
                    not in [2, 1 + len(self.data_obj.original_dim)]):
                raise ValueError('With stationary_observer=False,'
                                 'location_indices must be 2 or match\n'
                                 'self.data_obj.original_dim + 1')
            self.location_dim = np.array([a.shape[0] for a in
                                          self.location_indices])
            errors_vector = np.array([
                rng.normal(
                    loc=self.error_bias,
                    scale=self.error_sd,
                    size=ld)
                for ld in self.location_dim], dtype=object)

            if self.error_positive_only:
                errors_vector = np.array([
                    np.maximum(e, 0.) for e in errors_vector])

            values_vector = np.array([
                (self.data_obj.values[self.time_indices[i]]
                    [self.location_indices[i]] + errors_vector[i])
                for i in range(self.time_dim)], dtype=object)
            coords = self.location_indices

        return ObsVector(values=values_vector,
                         times=self.data_obj.times[self.time_indices],
                         coords=coords,
                         obs_dims=self.location_dim,
                         num_obs=values_vector.shape[0],
                         errors=errors_vector,
                         error_dist='normal'
                         )
