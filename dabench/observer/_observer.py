"""Base class for Observer object

Input is  generated data, returns ObsVector with values, times, coords, etc
"""

import numpy as np

from dabench.vector import ObsVector


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
        random_seed (int): Random seed for sampling times and locations
            according to location_density and time_density. Default is 99.
    """

    def __init__(self,
                 data_obj,
                 location_indices=None,
                 location_density=1.,
                 stationary_observers=True,
                 time_indices=None,
                 time_density=1.,
                 error_bias=0.,
                 error_sd=0.,
                 error_positive_only=False,
                 random_seed=99,
                 ):
        self.data_obj = data_obj
        if location_indices is not None:
            location_indices = np.array(location_indices)
        self.location_indices = location_indices
        self.location_density = location_density
        self.stationary_observers = stationary_observers
        if time_indices is not None:
            time_indices = np.array(time_indices)
        self.time_indices = time_indices
        self.time_density = time_density
        self.error_bias = error_bias
        self.error_sd = error_sd
        self.error_positive_only = error_positive_only
        self.random_seed = random_seed

    def _generate_time_indices(self, rng):
        self.time_indices = np.where(
                rng.binomial(1, p=self.time_density,
                             size=self.data_obj.time_dim
                             ).astype('bool')
                )[0]

    def _generate_stationary_indices(self, rng):
        self.location_indices = np.where(
                rng.binomial(1, p=self.location_density,
                             size=self.data_obj.system_dim
                             ).astype('bool')
                )[0]

    def _generate_nonstationary_indices(self, rng):
        self.location_indices = np.array([
                np.where(
                    rng.binomial(1, p=self.location_density,
                                 size=self.data_obj.system_dim
                                 ).astype('bool'))[0]
                for i in range(self.time_indices.shape[0])
                ], dtype=object)

    def _sample_stationary(self, errors_vector, sample_in_system_dim):
        if sample_in_system_dim:
            values_vector = (
                self.data_obj.values[self.time_indices][
                    :, self.location_indices]
                + errors_vector)
        else:
            # If sampling in gridded dimensions, need tuple for indexing
            tupled_inds = tuple(self.location_indices[:, i] for i in
                                range(self.location_indices.shape[1]))
            values_vector = np.array([
                self.data_obj.values_gridded[t][tupled_inds] for t in
                self.time_indices]) + errors_vector
        return values_vector

    def _sample_nonstationary(self, errors_vector, sample_in_system_dim):
        if sample_in_system_dim:
            values_vector = np.array([
                (self.data_obj.values[self.time_indices[i]]
                    [self.location_indices[i]] + errors_vector[i])
                for i in range(self.time_dim)], dtype=object)
        else:
            values_vector = np.array(
                [self.data_obj.values_gridded[self.time_indices[i]][
                    tuple(self.location_indices[i])]
                 + errors_vector[i] for i in range(self.time_dim)],
                dtype=object)
        return values_vector

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

        # Define random num generator
        rng = np.random.default_rng(self.random_seed)

        # Set time indices
        if self.time_indices is None:
            self._generate_time_indices(rng)

        self.time_dim = self.time_indices.shape[0]

        # For stationary observers (default)
        if self.stationary_observers:
            # Generate location_indices if not specified
            if self.location_indices is None:
                self._generate_stationary_indices(rng)

            # Check that location_indices are in correct dimensions
            if self.location_indices.shape[0] == 0:
                raise ValueError('location_indices is an empty list')
            elif len(self.location_indices.shape) == 1:
                sample_in_system_dim = True
            elif (self.location_indices.shape[1] ==
                    len(self.data_obj.original_dim)):
                sample_in_system_dim = False
            else:
                raise ValueError('location_indices must be 1D or match\n'
                                 'len(self.data_obj.original_dim)')

            # Generate errors
            self.location_dim = np.repeat(self.location_indices.shape[0],
                                          self.time_dim)
            errors_vec_size = (self.time_dim,) + (self.location_dim[0],)
            errors_vector = rng.normal(loc=self.error_bias,
                                       scale=self.error_sd,
                                       size=errors_vec_size)

            # Clip errors to positive only
            if self.error_positive_only:
                errors_vector[errors_vector < 0.] = 0.

            # Get values
            values_vector = self._sample_stationary(
                    errors_vector,
                    sample_in_system_dim)

            # Repeat location indices across time_dim for passing to ObsVector
            full_loc_indices = np.array(
                [self.location_indices] * self.time_dim)

        # If NON-stationary observer
        else:
            # Generate location_indices if not specified
            if self.location_indices is None:
                self._generate_nonstationary_indices(rng)

            # Check that location_indices are in correct dimensions
            if self.location_indices.shape[0] == 0:
                raise ValueError('location_indices is an empty list')
            elif len(self.location_indices[0].shape) == 1:
                sample_in_system_dim = True
            elif (self.location_indices[0].shape[1] ==
                  len(self.data_obj.original_dim)):
                sample_in_system_dim = False
            else:
                raise ValueError('With stationary_observers=False,'
                                 'location_indices must be 1D array of arrays,'
                                 ' with each element being 1D or matching\n'
                                 'self.data_obj.original_dim')
            self.location_dim = np.array([a.shape[0] for a in
                                          self.location_indices])

            # Generate errors
            errors_vector = np.array([
                rng.normal(
                    loc=self.error_bias,
                    scale=self.error_sd,
                    size=ld)
                for ld in self.location_dim], dtype=object)

            if self.error_positive_only:
                errors_vector = np.array([
                    np.maximum(e, 0.) for e in errors_vector])

            # Get values from generator
            values_vector = self._sample_nonstationary(
                    errors_vector,
                    sample_in_system_dim)

            # For passing to ObsVector
            full_loc_indices = self.location_indices

        return ObsVector(values=values_vector,
                         times=self.data_obj.times[self.time_indices],
                         time_indices=self.time_indices,
                         location_indices=full_loc_indices,
                         obs_dims=self.location_dim,
                         num_obs=values_vector.shape[0],
                         errors=errors_vector,
                         error_dist='normal'
                         )
