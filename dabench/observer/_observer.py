"""Base class for Observer object

Input is  generated data, returns ObsVector with values, times, coords, etc
"""

import warnings 

import numpy as np
import jax
import jax.numpy as jnp
import xarray as xr

from dabench.data import Data

# For typing
ArrayLike = np.ndarray | jax.Array


class Observer():
    """Flexibly samples observations from generated data

    Args:
        state_vec: Data generator/loader object from which
            to gather observations.
        random_location_density: Fraction of locations in
            system_dim to randomly select for observing, must be value
            between 0 and 1. Default is 1.
        random_time_density: Fraction of times to randomly select
            for observing must be value between 0 and 1. Default is 1.
        random_location_count: Number of locations in data_obj's
            system_dim to randomly select for observing. Default is None.
            User should specify one of: random_location_count,
            random_location_density, or location_indices.
            If random_location_count is specified, it takes precedent over
            random_location_density.
        random_time_count: Number of times to randomly select for
            observing. Default is None. User should specify one of:
            random_time_count, random_time_density, or time_indices.
            If random_time_count is specified, it takes precedent over
            random_time_density.
        locations: Manually specified indices for observing.
            If 1D array provided, assumed to be for flattened system_dim.
            If >1D, must have same dimensionality as data generator's
            original_dim (e.g. (x, y, z)). If stationary_observers=False,
            expects leading time dimension. If not specified, will be randomly
            generated according to random_location_density OR
            random_location_count. Default is None.
        times: Indices of times to gather observations from.
            If not specified, randomly generate according to
            random_time_density OR random_time_count. Default is None.
        stationary_observers: If True, samples from same indices at
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
        error_bias: Mean of normal distribution of
            observation errors. If provided as an array, it is taken to be
            variable-specific and the length must be equal to
            data_obj.system_dim. Default is 0.
        error_sd: Standard deviation of observation errors.
            observation errors. If provided as an array, it is taken to be
            variable-specific and the length be equal to data_obj.system_dim.
            Default is 0.
        error_positive_only: Clip errors to be positive only. Default is
            False.
        random_seed: Random seed for sampling times and locations.
            Default is 99.
        store_as_jax: Store values as jax array instead of numpy array.
            Default is False (store as numpy).

    Attributes:
        locations (ArrayLike): Location indices for making
            observations. In system_dim (1D) or original dim
            (>1D) of self.state_vec.
        location_dim (int): Number of locations sampled from (max
            in a single time step, if non-stationary observers).
        times (ArrayLike): Time indices to gather observations
            from.
        time_dim (int): Number of times sampled from.

    """

    def __init__(self,
                 state_vec: xr.Dataset,
                 random_time_density: float = 1.,
                 random_location_density: float | tuple[float, ...] = 1.,
                 random_time_count: int | None = None,
                 random_location_count: int | tuple[int, ...] | None = None,
                 times: ArrayLike | None = None,
                 locations: ArrayLike | None = None,
                 stationary_observers: bool = True,
                 error_bias: ArrayLike | float = 0.,
                 error_sd: ArrayLike | float = 0.,
                 error_positive_only: bool = False,
                 random_seed: int = 99,
                 store_as_jax: bool = False,
                 ):

        self.state_vec = state_vec
        self._coord_names = list(self.state_vec.coords.keys())
        self._nontime_coord_names = [coord for coord in self._coord_names
                                     if coord != 'time']
        self.state_vec = self.state_vec.assign_coords(
            {'variable': self.state_vec.data_vars}
            # 'variable_index': np.arange(len(self.state_vec.data_vars))}
        )
        # The system_index corresponds to the points location in a flattened 
        # array (i.e. state_vec[state_vec.data_vars].to_array().data.flatten())
        self.state_vec = self.state_vec.assign(
            {'system_index': (
                ['variable'] + ['time'] + self._nontime_coord_names,
                np.tile(
                    np.arange(self.state_vec.system_dim).reshape(
                        self.state_vec.sizes['variable'], -1
                        ),
                    self.state_vec.sizes['time']
                    ).reshape(self.state_vec.to_array().shape)
                       )
            }
        )

        if times is not None:
            times = np.array(times)
        self.times = times

        self.random_time_density = random_time_density
        self.random_time_count = random_time_count

        self.locations = locations
        self.random_location_density = random_location_density
        self.random_location_count = random_location_count
        self.stationary_observers = stationary_observers

        self.random_seed = random_seed
        if (store_as_jax and self.random_location_density != 1. and
                not self.stationary_observers):
            warnings.warn(
                "store_as_jax=True is not compatible with irregular"
                "observations (i.e. stationary_observers = False AND"
                "random_location_density != 1. Setting store_ax_jax"
                " to False and proceeding.")
            self.store_as_jax = False
        else:
            self.store_as_jax = store_as_jax


        self.error_bias = error_bias
        self.error_sd = error_sd

        if isinstance(self.error_bias, (list, np.ndarray, jnp.ndarray)):
            if len(self.error_bias) == 1:
                self._error_bias_is_list = False
            elif not len(self.error_bias) == self.state_vec.system_dim:
                raise ValueError(
                    "List of error biases has length {}."
                    "Must match system_dim ({}) or ".format(
                        len(self.error_bias), self.state_vec.system_dim))
            elif isinstance(self.error_bias, list):
                if self.store_as_jax:
                    self.error_bias = jnp.array(self.error_bias)
                else:
                    self.error_bias = np.array(self.error_bias)
            self._error_bias_is_list = True
        else:
            self._error_bias_is_list = False
                                 
        if isinstance(self.error_sd, (list, np.ndarray, jnp.ndarray)):
            if len(self.error_sd) == 1:
                self._error_sd_is_list = False
            elif not len(self.error_sd) == self.state_vec.system_dim:
                raise ValueError(
                    "List of error sds has length {}."
                    "Must match system_dim ({})".format(
                        len(self.error_sd), self.state_vec.system_dim))
            elif isinstance(self.error_sd, list):
                if self.store_as_jax:
                    self.error_sd = jnp.array(self.error_sd)
                else:
                    self.error_sd = np.array(self.error_sd)
            self._error_sd_is_list = True
        else:
            self._error_sd_is_list = False
                 
        self.error_positive_only = error_positive_only

    def _generate_times(
            self,
            rng: np.random.Generator
            ):
        if self.random_time_count is not None:
            self.times = np.sort(rng.choice(
                    self.state_vec['time'],
                    size=self.random_time_count,
                    replace=False,
                    shuffle=False))
        else:
            self.times = self.state_vec.time[np.where(
                    rng.binomial(1, p=self.random_time_density,
                                 size=self.state_vec.sizes['time']
                                 ).astype('bool')
                    )[0]]

    def _generate_stationary_locs(
            self,
            rng: np.random.Generator
            ):
        if self.random_location_count is not None:
            location_count = self.random_location_count
        else:
            location_count = np.sum(
                rng.binomial(1,
                             p=self.random_location_density,
                             size=self.state_vec.system_dim))
        if len(self._nontime_coord_names) > 1:
            sample_w_replace=True
        else:
            sample_w_replace=False
        self.locations = {
            coord_name: xr.DataArray(
                rng.choice(
                    self.state_vec[coord_name],
                    size=location_count,
                    replace=sample_w_replace,
                    shuffle=False),
                dims=['observations'])
            for coord_name in self._nontime_coord_names
        }
        self.location_dim = location_count

    def _generate_nonstationary_locs(
            self,
            rng: np.random.Generator
            ):
        """Generate different locations for each observation time"""
        if self.random_location_count is not None:
            self._location_counts = np.repeat(
                self.random_location_count, self.times.shape[0]
            )
        else:
            # An unequal amount of locations per time
            self._location_counts = [np.sum(
                rng.binomial(1,
                             p=self.random_location_density,
                             size=self.state_vec.system_dim)
                             )
            for i in range(self.times.shape[0])]

        if len(self._nontime_coord_names) > 1:
            sample_w_replace=True
        else:
            sample_w_replace=False

        self.locations = [{
            coord_name: xr.DataArray(
                rng.choice(
                    self.state_vec[coord_name],
                    size=lc,
                    replace=sample_w_replace,
                    shuffle=False),
                    dims=['observations'])
            for coord_name in self._nontime_coord_names
            }
        for lc in self._location_counts]

        self.location_dim = np.max(self._location_counts)

    def observe(self) -> xr.Dataset:
        """Generate observations.

        Returns:
            ObsVector containing observation values, times, locations, and
            errors
        """

        # Define random num generator
        rng = np.random.default_rng(self.random_seed)

        # Set time indices
        if self.times is None:
            self._generate_times(rng)

        self.time_dim = self.times.shape[0]

        # For stationary observers (default)
        if self.stationary_observers:
            # Generate locations if not specified
            if self.locations is None:
                self._generate_stationary_locs(rng)
            else:
                self.location_dim = next(iter(self.locations.items()))[1]['observations'].size


            # Sample
            obs_vec = self.state_vec.sel(time=self.times).sel(self.locations)

        # If NON-stationary observer
        else:
            # Generate location_indices if not specified
            if self.locations is None:
                self._generate_nonstationary_locs(rng)

            # If there's an unequal number of obs, will pad
            pad_widths = self.location_dim - np.array(self._location_counts)

            # Sample
            obs_vec = xr.concat([
                # Select by time
                self.state_vec.sel(
                        time=t
                # Select locations
                    ).sel(
                        self.locations[i]
                # Pad observations to max number
                    ).pad(
                        observations=(0, pad_widths[i])
                    )
                for i, t in enumerate(self.times)], 
                dim='time')

        # Transpose system_index to ensure consistency with flattened data
        obs_vec['system_index'] = obs_vec['system_index'].transpose('variable','time','observations').fillna(
            0).astype(int)

        # Generate errors
        errors_vec_size = ((self.time_dim,)
                           + (self.location_dim,)
                           + (obs_vec.sizes['variable'],))
        errors_vec_size = ((obs_vec.sizes['variable'],)
                           + (self.time_dim,)
                           + (self.location_dim,))
                      
        if self._error_bias_is_list:
            error_bias = self.error_bias[obs_vec['system_index'].data]
        else:
            error_bias = self.error_bias
        if self._error_sd_is_list:
            error_sd = self.error_sd[obs_vec['system_index'].data]
        else:
            error_sd = self.error_sd
        errors_vector = rng.normal(loc=error_bias,
                                    scale=error_sd,
                                    size=errors_vec_size)

        # Include flag for whether observations are stationary or not
        # Also include error_sd as an attribute
        obs_vec = obs_vec.assign_attrs(
            stationary_observers=self.stationary_observers,
            error_sd=error_sd)

        # Clip errors to positive only
        if self.error_positive_only:
            errors_vector[errors_vector < 0.] = 0.

        # Save errors and apply them to observations
        obs_vec = obs_vec.assign(errors=(obs_vec['system_index'].dims, errors_vector))
        for data_var in obs_vec['variable'].values:
            obs_vec[data_var] = obs_vec[data_var] + obs_vec['errors'].sel(variable=data_var)


        return obs_vec
