"""Interface with qgs to run quasigeostrpohic models

Requires qgs: https://qgs.readthedocs.io/

"""
import logging
import numpy as np
from copy import deepcopy
import jax.numpy as jnp

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

    def rhs(self, x, t):
        """Vector field (tendencies) of qgs system

        Arg:
            x (ndarray): State vector [size: (tstep, system_dim)]
        Returns:
            dx: vector field of qgs

        """

        dx = self.f(t, x)

        return dx

    def Jacobian(self, x, t):
        """Jacobian of the qgs system

        Arg:
            x (ndarray): State vector [size: (tstep, system_dim)]

        Returns:
            J (ndarray): Jacobian matrix, shape: (system_dim, system_dim)

        """

        J = self.Df(t, x)

        return J
