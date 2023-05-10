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
        random_location_density (float or tuple): Fraction of locations in
            system_dim to randomly select for observing, must be value
            between 0 and 1. Default is 1.
        random_time_density (float): Fraction of times to randomly select
            for observing must be value between 0 and 1. Default is 1.
        random_location_count (int): Number of locations in data_obj's
            system_dim to randomly select for observing. Default is None.
            User should specify one of: random_location_count,
            random_location_density, or location_indices.
            If random_location_count is specified, it takes precedent over
            random_location_density.
        random_time_count (int): Number of times to randomly select for
            observing. Default is None. User should specify one of:
            random_time_count, random_time_density, or time_indices. 
            If random_time_count is specified, it takes precedent over
            random_time_density.
        location_indices (ndarray): Manually specified indices for observing.
            If 1D array provided, assumed to be for flattened system_dim.
            If >1D, must have same dimensionality as data generator's
            original_dim (e.g. (x, y, z)). If stationary_observers=False,
            expects leading time dimension. If not specified, will be randomly
            generated according to random_location_density OR
            random_location_count. Default is None.
        time_indices (ndarray): Indices of times to gather observations from.
            If not specified, randomly generate according to
            random_time_density OR random_time_count. Default is None.
        stationary_observers (bool): If True, samples from same indices at 
            each time step. If False, randomly generates/expects new 
            observation indices at each timestep. Default is True.
            If False:
                If using random_location_count, the same number of indices
                    will be randomly generated. 
                If using random_location_density, indices are randomly
                    generated, with the possibility of a different number
                    of locations at each times step..
                If using location_indices, expects indices to either be 2D
                    (time_dim, system_dim) or >2D (time_dim, original_dim).
        error_bias (float): Mean of normal distribution of observation errors.
            Default is 0.
        error_sd (float): Standard deviation of observation errors. Default is
            0.
        error_positive_only (bool): Clip errors to be positive only. Default is
            False.
        random_seed (int): Random seed for sampling times and locations.
            Default is 99.
    """

    def __init__(self,
                 data_obj,
                 random_time_density=1.,
                 random_location_density=1.,
                 random_time_count=None,
                 random_location_count=None,
                 time_indices=None,
                 location_indices=None,
                 stationary_observers=True,
                 error_bias=0.,
                 error_sd=0.,
                 error_positive_only=False,
                 random_seed=99,
                 ):

        self.data_obj = data_obj

        if time_indices is not None:
            time_indices = np.array(time_indices)
        self.time_indices = time_indices
        self.random_time_density = random_time_density
        self.random_time_count = random_time_count

        if location_indices is not None:
            location_indices = np.array(location_indices)
        self.location_indices = location_indices
        self.random_location_density = random_location_density
        self.random_location_count = random_location_count
        self.stationary_observers = stationary_observers

        self.error_bias = error_bias
        self.error_sd = error_sd
        self.error_positive_only = error_positive_only
        self.random_seed = random_seed

    def _generate_time_indices(self, rng):
        if self.random_time_count is not None:
            self.time_indices = np.sort(rng.choice(
                    self.data_obj.time_dim,
                    size=self.random_time_count,
                    replace=False,
                    shuffle=False))
        else:
            self.time_indices = np.where(
                    rng.binomial(1, p=self.random_time_density,
                                 size=self.data_obj.time_dim
                                 ).astype('bool')
                    )[0]

    def _generate_stationary_indices(self, rng):
        if self.random_location_count is not None:
            self.location_indices = rng.choice(
                    self.data_obj.system_dim,
                    size=self.random_location_count,
                    replace=False,
                    shuffle=False)
        else:
            self.location_indices = np.where(
                    rng.binomial(1, p=self.random_location_density,
                                 size=self.data_obj.system_dim
                                 ).astype('bool')
                    )[0]

    def _generate_nonstationary_indices(self, rng):
        if self.random_location_count is not None:
            self.location_indices = np.array([
                rng.choice(
                    self.data_obj.system_dim,
                    size=self.random_location_count,
                    replace=False,
                    shuffle=False)
                for i in range(self.time_indices.shape[0])])
        else:
            self.location_indices = np.array([
                    np.where(
                        rng.binomial(1, p=self.random_location_density,
                                     size=self.data_obj.system_dim
                                     ).astype('bool'))[0]
                    for i in range(self.time_indices.shape[0])
                    ], dtype=object)

    def _generate_stationary_indices_gridded(self, rng):
        if self.random_location_count is not None:
            arange_list = [np.arange(n) for n in self.data_obj.original_dim]
            ind_possibilities = np.array(
                np.meshgrid(*arange_list)).T.reshape(
                    -1, len(self.data_obj.original_dim))
            self.location_indices = rng.choice(
                    ind_possibilities,
                    size=self.random_location_count,
                    replace=False,
                    shuffle=False)
        else:
            self.location_indices = np.array(np.where(
                    rng.binomial(1, p=self.random_location_density,
                                 size=self.data_obj.original_dim
                                 ).astype('bool')
                    )).T

    def _generate_nonstationary_indices_gridded(self, rng):
        if self.random_location_count is not None:
            arange_list = [np.arange(n) for n in self.data_obj.original_dim]
            ind_possibilities = np.array(
                np.meshgrid(*arange_list)).T.reshape(
                    -1, len(self.data_obj.original_dim))
            self.location_indices = np.array([rng.choice(
                    ind_possibilities,
                    size=self.random_location_count,
                    replace=False,
                    shuffle=False) for i in range(self.time_indices.shape[0])])
        else:
            self.location_indices = np.array([
                    np.array(np.where(
                        rng.binomial(1, p=self.random_location_density,
                                     size=self.data_obj.original_dim
                                     ).astype('bool'))).T
                    for i in range(self.time_indices.shape[0])
                    ], dtype=object)

    def _sample_stationary(self, errors_vector, sample_in_system_dim):
        if sample_in_system_dim:
            values_vector = (
                self.data_obj.values[self.time_indices][
                    :, self.location_indices]
                + errors_vector)
        else:
            values_vector = np.array([
                self.data_obj.values_gridded[t][tuple(self.location_indices.T)]
                for t in self.time_indices]) + errors_vector
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
                    tuple(self.location_indices[i].T)]
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
                # Check if data is in spectral or physical space
                if (hasattr(self.data_obj, 'is_spectral') and
                        self.data_obj.is_spectral):
                    self._generate_stationary_indices_gridded(rng)
                else:
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
                # Check if data is in spectral or physical space
                if (hasattr(self.data_obj, 'is_spectral') and
                        self.data_obj.is_spectral):
                    self._generate_nonstationary_indices_gridded(rng)
                else:
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
                         error_dist='normal',
                         error_sd=self.error_sd,
                         error_bias=self.error_bias
                         )
