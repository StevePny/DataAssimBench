"""Interface with dinosaur package to run models

Requires dinosaur: 

See NeuralGCM for more info also: 

"""

import logging
import numpy as np
from copy import deepcopy
import jax.numpy as jnp
import functools
import xarray
import jax

from dabench.data import _data

logging.basicConfig(filename='logfile.log', level=logging.DEBUG)

try:
    import dinosaur
except ImportError:
    dinosaur = None
    logging.warning(
        'Package: dinosaur not found!\n'
        'data.Dinosaur will not work without this optional package\n'
        'To install via pip: '
        'pip install git+https://github.com/google-research/dinosaur\n'
        'For more information: https://github.com/google-research/dinosaur'
        )


class Dinosaur(_data.Data):
    """Class to set up Dinosaur global atmospheric model

    The Dinosaur class is simply a wrapper of a "optional" pyqg package.
    See: https://github.com/google-researchjj/dinosaur
    Also: https://neuralgcm.readthedocs.io/en/latest/installation.html 

    Attributes:
        config (string): Type of model to run. Right now, only supports
            'baroclinic'. Default is 'baroclinic'.
        grid_abbreviation (string): Spherical harmonic grid configuration. See
            https://climatedataguide.ucar.edu/climate-tools/common-spectral-model-grid-resolutions
            for examples. Default is 'T42'. Consider using 'T85' for higher
            resolution.
        layers (int): Number of model levels. Default is 25.
        delta_t (float): Numerical timestep. Units: seconds. Default is 100.
        store_as_jax (bool): Store values as jax array instead of numpy array.
            Default is True for dinosaur (store as numpy).
    """
    def __init__(self,
                 config='baroclinic',
                 grid_abbreviation='T42',
                 layers=25,
                 delta_t=100,
                 system_dim=None,
                 x0=None,
                 time_dim=None,
                 values=None,
                 times=None,
                 store_as_jax=True,
                 **kwargs):
        """ Initialize Dinosaur data object"""

        self.store_as_jax = store_as_jax
        self.units = dinosaur.scales.units
        self.config = config
        self.grid_abbreviation = grid_abbreviation
        self.grid = getattr(dinosaur.spherical_harmonic.Grid,
                            self.grid_abbreviation)()
        self.vertical_grid = dinosaur.sigma_coordinates.SigmaCoordinates.equidistant(layers)
        self.coords = dinosaur.coordinate_systems.CoordinateSystem(
            self.grid, self.vertical_grid)
        self.physics_specs = dinosaur.primitive_equations.PrimitiveEquationsSpecs.from_si()
        self.initial_state_fn, self.aux_features = dinosaur.primitive_equations_states.steady_state_jw(
            self.coords, self.physics_specs)
        self.steady_state = self.initial_state_fn()
        self.ref_temps = self.aux_features[dinosaur.xarray_utils.REF_TEMP_KEY]
        self.orography = dinosaur.primitive_equations.truncated_modal_orography(
            self.aux_features[dinosaur.xarray_utils.OROGRAPHY], self.coords)
        self.delta_t = delta_t * self.units.s
        self._delta_t_nondim = self.physics_specs.nondimensionalize(self.delta_t)

        # Setting up step function
        self.primitive = dinosaur.primitive_equations.PrimitiveEquations(
            self.ref_temps, self.orography, self.coords, self.physics_specs)
        temp_step_fn = dinosaur.time_integration.imex_rk_sil3(
            self.primitive, self._delta_t_nondim) 
        filters = [dinosaur.time_integration.exponential_step_filter(self.grid, self._delta_t_nondim),]
        self.step_fn = dinosaur.time_integration.step_with_filters(temp_step_fn, filters)

        if self.config == 'baroclinic':
            self.perturbation = dinosaur.primitive_equations_states.baroclinic_perturbation_jw(
                 self.coords, self.physics_specs)
            self._x0_dino = self.steady_state + self.perturbation
        else:
            raise ValueError('Not a valid model configuration. '
                             'Must be one of: "baroclinic"')

        super().__init__(system_dim=system_dim, time_dim=time_dim,
                         values=values, times=times, delta_t=delta_t,
                         store_as_jax=store_as_jax, x0=x0,
                         **kwargs)


    def dimensionalize(self, x, unit):
        """Dimensionalizes `xarray.DataArray`s.
        
        Args:
            x (xarray.DataArray): DataArray to dimensionalize.
            unit (dinosaur.scales.units.Unit): Unit for dimensionalization.

        Returns:
            xarray.DataArray containing dimensionalized x.
        """
        dimensionalize = functools.partial(self.physics_specs.dimensionalize, unit=unit)
        return xarray.apply_ufunc(dimensionalize, x)

    def generate(self, n_steps=None, t_final=None, x0=None,
                 save_every_n_steps=1):
        """Generates values and times, saves them to the data object

        Notes:
            Either provide n_steps or t_final in order to indicate the length
            of the forecast. These are used to set the values, times, and
            time_dim attributes.

        Args:
            n_steps (int): Number of timesteps. One of n_steps OR
                t_final must be specified.
            t_final (float): Final time of trajectory. One of n_steps OR
                t_final must be specified.
            x0 (ndarray, optional): the initial conditions. Can also be
                provided when initializing model object. If provided by
                both, the generate() arg takes precedence.
            save_every_n_steps (int, optional): Number of timesteps between
                saved outputs. Default is 1 (i.e. every time step is saved).
        """

        # Set seed
        np.random.seed(37)

        # Checks
        # Check that n_steps or t_final is supplied
        if n_steps is not None:
            t_final = n_steps * self.delta_t
        elif t_final is not None:
            n_steps = int(t_final/self.delta_t)
        else:
            raise TypeError('Either n_steps or t_final must be supplied as an '
                            'input argument.')
        save_every_t = save_every_n_steps*self.delta_t

        inner_steps = int(save_every_t / self.delta_t)
        outer_steps = int(t_final / save_every_t)
        integrate_fn = dinosaur.time_integration.trajectory_from_step(
            self.step_fn, outer_steps, inner_steps)
        integrate_fn = jax.jit(integrate_fn)
        final, trajectory = jax.block_until_ready(integrate_fn(self._x0_dino))
        self.trajectory = jax.device_get(trajectory)
        self.times = (save_every_t * np.arange(outer_steps))

        # Formatting predictions to xarray and updating attributes
        trajectory = jax.device_get(trajectory)

        trajectory_dict, _ = dinosaur.pytree_utils.as_dict(trajectory)
        u, v = dinosaur.spherical_harmonic.vor_div_to_uv_nodal(
            self.grid, trajectory.vorticity, trajectory.divergence)
        trajectory_dict.update({'u': u, 'v': v})
        nodal_trajectory_fields = dinosaur.coordinate_systems.maybe_to_nodal(
            trajectory_dict, coords=self.coords)
        trajectory_ds = dinosaur.xarray_utils.data_to_xarray(
            nodal_trajectory_fields, coords=self.coords, times=self.times)

        trajectory_ds['surface_pressure'] = np.exp(trajectory_ds.log_surface_pressure[:, 0, :,:])
        temperature = dinosaur.xarray_utils.temperature_variation_to_absolute(
            trajectory_ds.temperature_variation.data, self.ref_temps)
        trajectory_ds = trajectory_ds.assign(
            temperature=(trajectory_ds.temperature_variation.dims, temperature))
        self._import_xarray_ds(trajectory_ds, exclude_vars=['surface_pressure', 'log_surface_pressure'])


    def forecast(self, n_steps=None, t_final=None, x0=None):
        """Alias for self.generate(), except returns values as output"""
        self.generate(n_steps, t_final, x0)

        return self.values