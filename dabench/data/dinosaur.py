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
            'baroclinic' and 'held-suarez'. Default is 'baroclinic'.
        grid_abbreviation (string): Spherical harmonic grid configuration. See
            https://climatedataguide.ucar.edu/climate-tools/common-spectral-model-grid-resolutions
            for examples. Default is 'T42'. Consider using 'T85' for higher
            resolution.
        layers (int): Number of model levels. Default is 25.
    """
    def __init__(self,
                 config='baroclinic',
                 grid_abbreviation='T42',
                 layers=25,
                 time_dim=None,
                 values=None,
                 times=None,
                 store_as_jax=False,
                 **kwargs):
        """ Initialize Dinosaur data object"""

        self.config = config
        self.grid_abbreviation = grid_abbreviation
        self.grid = getattr(dinosaur.spherical_harmonic.Grid,
                            self.grid_abbrevation)()
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

        if self.config == 'baroclinic':
            self.perturbation = dinosaur.primitive_equations_states.baroclinic_perturbation_jw(
                 self.coords, selfphysics_specs)
            self.x0 = self.steady_state + self.perturbation




        self.system_dim = self.x0.size
        # super().__init__(system_dim=system_dim, time_dim=time_dim,
        #                  values=values, times=times, delta_t=delta_t,
        #                  store_as_jax=store_as_jax, x0=x0,
        #                  **kwargs)
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

    def generate(self, n_steps=None, t_final=None, x0=None):
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


    def forecast(self, n_steps=None, t_final=None, x0=None):
        """Alias for self.generate(), except returns values as output"""
        self.generate(n_steps, t_final, x0)

        return self.values