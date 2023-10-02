"""Interface with qgs to run quasigeostrpohic models

Requires qgs: https://qgs.readthedocs.io/

"""
import logging
import numpy as np
from copy import deepcopy
from dabench.data._utils import integrate

from dabench.data import _data

logging.basicConfig(filename='logfile.log', level=logging.DEBUG)

try:
    from qgs.params.params import QgParams
    from qgs.functions.tendencies import create_tendencies
    qgs = True
except ImportError:
    qgs = None
    logging.warning(
        'Package: qgs not found!\n'
        'QGS data model will not work without this optional package\n'
        'To install: pip install qgs\n'
        'For more information: https://qgs.readthedocs.io/en/latest/files/general_information.html'
        )


class QGS(_data.Data):
    """ Class to set up QGS quasi-geostrophic model

    The QGS class is simply a wrapper of an *optional* qgs package.
    See https://qgs.readthedocs.io/

    Attributes:
        model_params (QgParams): qgs parameter object. See:
            https://qgs.readthedocs.io/en/latest/files/technical/configuration.html#qgs.params.params.QgParams
            If None, will use defaults specified by:
            De Cruz, et al. (2016). Geosci. Model Dev., 9, 2793-2808.
        delta_t (float): Numerical timestep. Units: seconds.
        store_as_jax (bool): Store values as jax array instead of numpy array.
            Default is False (store as numpy).
        x0 (ndarray): Initial state vector, array of floats. Default is:
    """
    def __init__(self,
                 model_params=None,
                 x0=None,
                 delta_t=0.1,
                 system_dim=None, 
                 time_dim=None,
                 values=None,
                 times=None,
                 store_as_jax=False,
                 random_seed=37,
                 **kwargs):
        """ Initialize qgs object, subclass of Base

        See: https://qgs.readthedocs.io/"""

        if qgs is None:
            raise ModuleNotFoundError(
                'Package: qgs not found!\n'
                'QGS data model will not work without this optional package\n'
                'To install: pip install qgs\n'
                'For more information: https://qgs.readthedocs.io/en/latest/files/general_information.html'
                )

        if model_params is None:
            self.model_params = self._create_default_qgparams()
        self.random_seed = random_seed
        self._rng = np.random.default_rng(self.random_seed)

        if system_dim is None:
            system_dim = self.model_params.ndim
        elif system_dim != self.model_params.ndim:
            print('WARNING: input system_dim is ' + str(system_dim)
                  + ' , setting system_dim = ' + str(self.model_params.ndim) + '.')
            system_dim = self.model_params.ndim

        if x0 is None:
            x0 = self._rng.random(system_dim)*0.001

        super().__init__(system_dim=system_dim, time_dim=time_dim,
                         values=values, times=times, delta_t=delta_t,
                         store_as_jax=store_as_jax, x0=x0,
                         **kwargs)

        self.f, self.Df = create_tendencies(self.model_params)

    def _create_default_qgparams(self):
        model_params = QgParams()

        # Mode truncation at the wavenumber 2 in both x and y spatial
        # coordinates for the atmosphere
        model_params.set_atmospheric_channel_fourier_modes(2, 2)
        # Mode truncation at the wavenumber 2 in the x and at the
        # wavenumber 4 in the y spatial coordinates for the ocean
        model_params.set_oceanic_basin_fourier_modes(2, 4)

        # Setting MAOOAM parameters according to
        # De Cruz, L., Demaeyer, J. and Vannitsem, S. (2016). 
        # Geosci. Model Dev., 9, 2793-2808.
        model_params.set_params({'kd': 0.0290, 'kdp': 0.0290, 'n': 1.5,
                                 'r': 1.e-7, 'h': 136.5, 'd': 1.1e-7})
        model_params.atemperature_params.set_params({'eps': 0.7, 'T0': 289.3,
                                                     'hlambda': 15.06})
        model_params.gotemperature_params.set_params({'gamma': 5.6e8,
                                                      'T0': 301.46})

        # Setting the short-wave radiation component:
        model_params.atemperature_params.set_insolation(103.3333, 0)
        model_params.gotemperature_params.set_insolation(310, 0)

        return model_params

    def rhs(self, x, t=0):
        """Vector field (tendencies) of qgs system

        Arg:
            x (ndarray): State vector, shape: (system_dim)
            t: times vector. Required as argument slot for some numerical
                integrators but unused.
        Returns:
            dx: vector field of qgs

        """

        dx = self.f(t, x)

        return dx

    def Jacobian(self, x, t=0):
        """Jacobian of the qgs system

        Arg:
            x (ndarray): State vector, shape: (system_dim)
            t: times vector. Required as argument slot for some numerical
                integrators but unused.

        Returns:
            J (ndarray): Jacobian matrix, shape: (system_dim, system_dim)

        """

        J = self.Df(t, x)

        return J

    def generate(self, n_steps=None, t_final=None, x0=None, M0=None,
                 return_tlm=False, stride=None, **kwargs):
        """Generates a dataset and assigns values and times to the data object.

        Notes:
            Either provide n_steps or t_final in order to indicate the length
            of the forecast. These are used to set the values, times, and
            time_dim attributes.

        Args:
            n_steps (int): Number of timesteps. One of n_steps OR
                t_final must be specified.
            t_final (float): Final time of trajectory. One of n_steps OR
                t_final must be specified.
            M0 (ndarray): the initial condition of the TLM matrix computation
                shape (system_dim, system_dim).
            return_tlm (bool): specifies whether to compute and return the
                integrated Jacobian as a Tangent Linear Model for each
                timestep.
            x0 (ndarray): initial conditions state vector of shape (system_dim)
            stride (int): specify how many steps to skip in the output data
                versus the model timestep (delta_t).
            **kwargs: arguments to the integrate function (permits changes in
                convergence tolerance, etc.).

        Returns:
            Nothing if return_tlm=False. If return_tlm=True, a list
                of TLMs corresponding to the system trajectory.
        """

        # Check that n_steps or t_final is supplied
        if n_steps is not None:
            t_final = n_steps * self.delta_t
        elif t_final is not None:
            n_steps = int(t_final/self.delta_t)
        else:
            raise TypeError('Either n_steps or t_final must be supplied as an '
                            'input argument.')

        # Check that x0 initial conditions is supplied
        if x0 is None:
            if self.x0 is not None:
                x0 = self.x0
            else:
                raise TypeError('Initial condition is None, x0 = {}. it must '
                                'either be provided as an argument or set as '
                                'an attribute in the model object.'.format(x0))

        # Check that self.rhs is defined
        if self.rhs is None:
            raise AttributeError('self.rhs must be specified prior to '
                                 'calling generate.')

        # Set f matrix for integrate
        if return_tlm:
            # Prep x0
            if M0 is None:
                M0 = np.identity(self.system_dim)
            xaux0 = np.concatenate((x0.ravel(), M0.ravel()))
            x0 = xaux0
            # Check that self.rhs_aux is defined
            if self.rhs_aux is None:
                raise AttributeError('self.rhs_aux must be specified prior to '
                                     'calling generate.')
            f = self.rhs_aux
        else:
            f = self.rhs

        # Integrate and store values and times
        # If data object has its own integration method, use that
        if hasattr(self, 'integrate') and callable(getattr(self, 'integrate')):
            y, t = self.integrate(f, x0, t_final, self.delta_t, stride=stride,
                                  **kwargs)
        # Otherwise, use integrate from dabench.support.utils
        else:
            y, t = integrate(f, x0, t_final, self.delta_t, stride=stride,
                             jax_comps=self.store_as_jax,
                             **kwargs)

        # The generate method specifically stores data in the object,
        # as opposed to the forecast method, which does not.
        # Store values and times as part of data object
        self.values = y[:, :self.system_dim]
        self.times = t
        self.time_dim = len(t)

        # Return the data series and associated TLMs if requested
        if return_tlm:
            # Reshape M matrix
            M = np.reshape(y[:, self.system_dim:],
                           (self.time_dim,
                            self.system_dim,
                            self.system_dim)
                           )

            if self.store_as_jax:
                return M
            else:
                return np.array(M)

    def _import_xarray_ds(self, ds, include_vars=None, exclude_vars=None,
                          years_select=None, dates_select=None,
                          lat_sorting=None):
        if dates_select is not None:
            dates_filter_indices = ds.time.dt.date.isin(dates_select)
            # First check to make sure the dates exist in the object
            if dates_filter_indices.sum() == 0:
                raise ValueError('Dataset does not contain any of the dates'
                                 ' specified in dates_select\n'
                                 'dates_select = {}\n'
                                 'NetCDF contains {}'.format(
                                     dates_select,
                                     np.unique(ds.time.dt.date)
                                     )
                                 )
            else:
                ds = ds.isel(time=dates_filter_indices)
        else:
            if years_select is not None:
                year_filter_indices = ds.time.dt.year.isin(years_select)
                # First check to make sure the years exist in the object
                if year_filter_indices.sum() == 0:
                    raise ValueError('Dataset does not contain any of the '
                                     'years specified in years_select\n'
                                     'years_select = {}\n'
                                     'NetCDF contains {}'.format(
                                         years_select,
                                         np.unique(ds.time.dt.year)
                                         )
                                     )
                else:
                    ds = ds.isel(time=year_filter_indices)

        # Check size before loading
        size_gb = ds.nbytes / (1024 ** 3)
        if size_gb > 1:
            warnings.warn('Trying to load large xarray dataset into memory. \n'
                          'Size: {} GB. Operation may take a long time, '
                          'stall, or crash.'.format(size_gb))

        # Get dims
        dims = ds.dims
        dims_names = list(ds.dims)

        # Set times
        time_key = None
        dims_keys = dims.keys()
        if 'time' in dims_keys:
            time_key = 'time'
        elif 'times' in dims_keys:
            time_key = 'times'
        elif 'time0' in dims_keys:
            time_key = 'time0'
        if time_key is not None:
            self.times = ds[time_key].values
            self.time_dim = self.times.shape[0]
        else:
            self.times = np.array([0])
            self.time_dim = 1

        # Find names for key dimensions: lat, lon, level (if it exists)
        lat_key = None
        lon_key = None
        lev_key = None
        if 'level' in dims_keys:
            lev_key = 'level'
        elif 'lev' in dims_keys:
            lev_key = 'lev'
        if 'latitude' in dims_keys:
            lat_key = 'latitude'
        elif 'lat' in dims_keys:
            lat_key = 'lat'
        if 'longitude' in dims_keys:
            lon_key = 'longitude'
        elif 'lon' in dims_keys:
            lon_key = 'lon'

        # Reorder dimensions: time, level, lat, lon, etc.
        dim_order = np.array([time_key, lev_key, lat_key, lon_key])
        dim_order = dim_order[dim_order != np.array(None)]
        remaining_dims = [d for d in dims_names if d not in dim_order]
        full_dim_order = list(dim_order) + remaining_dims

        if len(full_dim_order) > 0:
            ds = ds.transpose(*full_dim_order)

        # Orient data vertically
        if lat_key is not None:
            if lat_sorting is not None:
                if lat_sorting == 'ascending':
                    ds = ds.sortby(lat_key, ascending=True)
                elif lat_sorting == 'descending':
                    ds = ds.sortby(lat_key, ascending=False)
                else:
                    warnings.warn('{} is not a valid value for lat_sorting.\n'
                                  'Choose one of None, "ascending", or '
                                  '"descending".\n'
                                  'Proceeding without sorting.'.format(
                                      lat_sorting)
                                  )
        #  Get variable names and shapes
        names_list = []
        shapes_list = []
        if exclude_vars is not None:
            ds = ds.drop_vars(exclude_vars)
        if include_vars is not None:
            ds = ds[include_vars]
        for data_var in ds.data_vars:
            shapes_list.append(ds[data_var].shape)
            names_list.append(data_var)

        # Check if all elements' data shapes are equal
        if len(names_list) == 0:
            raise ValueError('No valid data_vars were found in dataset.\n'
                             'Check your include_vars and exclude_vars args.')
        if not shapes_list.count(shapes_list[0]) == len(shapes_list):
            # Formatting for showing variable names and shapes
            var_shape_warn_list = ['{:<12} {:<15}'.format(
                    'Variable', 'Dimensions')]
            var_shape_warn_list += ['{:<16} {:<16}'.format(
                names_list[i], str(shapes_list[i]))
                for i in range(len(shapes_list))]
            warnings.warn('data_vars do not all share the same dimensions.\n'
                          'Broadcasting variables to same dimensions.\n'
                          'To avoid, use include_vars or exclude_vars args.\n'
                          'Variable dimensions are:\n'
                          '{}'.format('\n'.join(var_shape_warn_list))
                          )

        # Gather values and set dimensions
        temp_values = np.moveaxis(np.array(ds.to_array()), 0, -1)
        self.original_dim = temp_values.shape[1:]
        if self.original_dim[-1] == 1 and len(self.original_dim) > 2:
            self.original_dim = self.original_dim[:-1]

        self.values = temp_values.reshape(
                temp_values.shape[0], -1)
        self.var_names = np.array(names_list)
        if self.x0 is None:
            self.x0 = self.values[0]
        self.time_dim = self.values.shape[0]
        self.system_dim = self.values.shape[1]
        if len(full_dim_order) == 0:
            warnings.warn('Unable to find any spatial or level dimensions '
                          'in dataset. Setting original_dim to system_dim: '
                          '{}'.format(self.system_dim))

    def load_netcdf(self, filepath=None, include_vars=None, exclude_vars=None,
                    years_select=None, dates_select=None,
                    lat_sorting='descending'):
        """Loads values from netCDF file, saves them in values attribute

        Args:
            filepath (str): Path to netCDF file to load. If not given,
                defaults to loading ERA5 ECMWF SLP data over Japan
                from 2018 to 2021.
            include_vars (list-like): Data variables to load from NetCDF. If
                None (default), loads all variables. Can be used to exclude bad
                variables.
            exclude_vars (list-like): Data variabes to exclude from NetCDF
                loading. If None (default), loads all vars (or only those
                specified in include_vars). It's recommended to only specify
                include_vars OR exclude_vars (unless you want to do extra
                typing).
            years_select (list-like): Years to load (ints). If None, loads all
                timesteps.
            dates_select (list-like): Dates to load. Elements must be
                datetime date or datetime objects, depending on type of time
                indices in NetCDF. If both years_select and dates_select
                are specified, time_stamps overwrites "years" argument. If
                None, loads all timesteps.
            lat_sorting (str): Orient data by latitude:
                descending (default), ascending, or None (uses orientation
                from data file).
        """
        if filepath is None:
            # Use importlib.resources to get the default netCDF from dabench
            with resources.open_binary(
                    _suppl_data, 'era5_japan_slp.nc') as nc_file:
                with xr.open_dataset(nc_file, decode_coords='all') as ds:
                    self._import_xarray_ds(
                        ds, include_vars=include_vars,
                        exclude_vars=exclude_vars,
                        years_select=years_select, dates_select=dates_select,
                        lat_sorting=lat_sorting)
        else:
            with xr.open_dataset(filepath, decode_coords='all') as ds:
                self._import_xarray_ds(
                    ds, include_vars=include_vars,
                    exclude_vars=exclude_vars,
                    years_select=years_select, dates_select=dates_select,
                    lat_sorting=lat_sorting)

    def save_netcdf(self, filename):
        """Saves values in values attribute to netCDF file

        Args:
            filepath (str): Path to netCDF file to save
        """

        # Set variable names
        if not hasattr(self, 'var_names') or self.var_names is None:
            var_names = ['var{}'.format(i) for
                         i in range(self.values.shape[1])]
        else:
            var_names = self.var_names

        # Set times
        if not hasattr(self, 'times') or self.times is None:
            times = np.arange(self.values.shape[0])
        else:
            times = self.times

        # Get values as list:
        values_list = [('time', self.values[:, i]) for i in range(
            self.values.shape[1])]

        data_dict = dict(zip(var_names, values_list))
        coords_dict = {
            'time': times,
            'system_dim': range(len(var_names))
            }
        ds = xr.Dataset(
            data_vars=data_dict,
            coords=coords_dict
            )

        ds.to_netcdf(filename, mode='w')

    def rhs_aux(self, x, t):
        """The auxiliary model used to compute the TLM.

        Args:
          x (ndarray): State vector with size (system_dim)
          t (ndarray): Array of times with size (time_dim)

        Returns:
          dxaux (ndarray): State vector [size: (system_dim,)]

        """
        # Compute M
        dxdt = self.rhs(x[:self.system_dim], t)
        J = self.Jacobian(x[:self.system_dim])
        M = np.array(np.reshape(x[self.system_dim:], (self.system_dim,
                                                      self.system_dim)))

        # Matrix multiplication
        dM = J @ M

        dxaux = np.concatenate((dxdt, dM.flatten()))

        return dxaux

    def calc_lyapunov_exponents_series(self, total_time=None, rescale_time=1,
                                       convergence=0.01, x0=None):
        """Computes the spectrum of Lyapunov Exponents.

        Notes:
            Lyapunov exponents help describe the degree of "chaos" in the
            model. Make sure to plot the output to check that the algorithm
            converges. There are three ways to make the estimate more accurate:
                1. Decrease the delta_t of the model
                2. Increase total_time
                3. Decrease rescale time (try increasing total_time first)
            Algorithm: Eckmann 85,
            https://www.ihes.fr/~ruelle/PUBLICATIONS/%5B81%5D.pdf pg 651
            This method computes the full time series of Lyapunov Exponents,
            which is useful for plotting for debugging. To get only the final
            Lyapunov Exponent, use self.calc_lyapunov_exponents.

        Args:
            total_time (float) : Time to integrate over to compute LEs.
                Usually there's a tradeoff between accuracy and computation
                time (more total_time leads to higher accuracy but more
                computation time). Default depends on model type and are based
                roughly on how long it takes for satisfactory convergence:
                For Lorenz63: n_steps=15000 (total_time=150 for delta_t=0.01)
                For Lorenz96: n_steps=50000 (total_time=500 for delta_t=0.01)
            rescale_time (float) : Time for when the algorithm rescales the
                propagator to reduce the exponential growth in errors.
                Default is 1 (i.e. 100 timesteps when delta_t = 0.01).
            convergence (float) : Prints warning if LE convergence is below
                this number. Default is 0.01.
            x0 (array) : initial condition to start computing LE.  Needs
                to be on the attractor (i.e., remove transients). Default is
                None, which will fallback to use the x0 set during model object
                initialization.

        Returns:
            Lyapunov exponents for all timesteps, array of size
                (total_time/rescale_time - 1, system_dim)
        """

        # Set total_time
        if total_time is None:
            subclass_name = self.__class__.__name__
            if subclass_name == 'Lorenz63':
                total_time = int(15000*self.delta_t)
            elif subclass_name == 'Lorenz96':
                total_time = int(50000*self.delta_t)
            else:
                total_time = 100

        if rescale_time > total_time:
            raise ValueError('rescale_time must be less than or equal to '
                             'total_time. Current values are rescale_time = {}'
                             ' and total_time = {}'.format(rescale_time,
                                                           total_time))
        D = self.system_dim
        times = np.arange(0, total_time, rescale_time)

        # Array to be populated with Lyapunov Exponents
        LE = np.zeros((len(times)-1, D))

        # Set initial conditions for first time period
        M0 = np.eye(D)
        prev_R = np.zeros(D)

        # Loop over rescale time periods
        for i, (t1, t2) in enumerate(zip(times[:-1], times[1:])):

            M = self.generate(t_final=t2-t1, x0=x0, M0=M0, return_tlm=True)
            x_t2 = self.values[-1]
            M_t2 = M[-1]

            Q, R = np.linalg.qr(M_t2)

            curr_R = np.log(np.abs(np.diag(R)))
            LE[i] = (prev_R+curr_R)/t2
            prev_R += curr_R

            x0 = x_t2
            M0 = Q
        # Check convergence
        compute_conv = np.mean(np.abs(LE[-11:-1] - LE[-10:]), axis=0)
        if np.any(compute_conv > convergence):
            print('WARNING: Exceeding convergence = {} > {}. Increase total_'
                  'time and/or decrease rescale_time.'.format(compute_conv,
                                                              convergence))

        return LE
