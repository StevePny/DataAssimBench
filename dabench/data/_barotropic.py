"""Interface with pyqg to run barotropic models

Requires pyqg: https://pyqg.readthedocs.io/

"""
import logging
import numpy as np
from copy import deepcopy
import jax
import jax.numpy as jnp
import xarray as xr

from dabench.data import _data

logging.basicConfig(filename='logfile.log', level=logging.DEBUG)

try:
    import pyqg
except ImportError:
    pyqg = None
    logging.warning(
        'Package: pyqg not found!\n'
        'Barotropic will not work without this optional package\n'
        'To install via conda: conda install -c conda-forge pyqg\n'
        'For more information: https://pyqg.readthedocs.io/en/latest/installation.html'
        )

# For typing
ArrayLike = np.ndarray | jax.Array


class Barotropic(_data.Data):
    """Barotropic model data generator based on pyqg

    This data class is a wrapper of a "optional" pyqg package.
    See https://pyqg.readthedocs.io

    Notes:
        DEPRECATED
        Uses default attribute values from pyqg.BTModel:
        https://pyqg.readthedocs.io/en/latest/api.html#pyqg.BTModel
        Those values originally come from Mcwilliams 1984:
            J. C. Mcwilliams (1984). The emergence of isolated coherent
            vortices in turbulent flow. Journal of Fluid Mechanics, 146,
            pp 21-43 doi:10.1017/S0022112084001750.

    Args:
        system_dim: system dimension
        beta: Gradient of coriolis parameter. Units: meters^-1 *
            seconds^-1. Default is 0.
        rek: Linear drag in lower layer. Units: seconds^-1.
            Default is 0.
        rd: Deformation radius. Units: meters. Default is 0.
        H: Layer thickness. Units: meters. Default is 1.
        nx: Number of grid points in the x direction. Default is 256.
        ny: Number of grid points in the y direction. Default: nx.
        L: Domain length in x direction. Units: meters. Default is
            2*pi.
        x0: the initial conditions. Can also be
            provided when initializing model object. If provided by
            both, the generate() arg takes precedence.
        W: Domain width in y direction. Units: meters. Default: L.
        filterfac (float): amplitdue of the spectral spherical filter.
            Default is 23.6.
        delta_t: Numerical timestep. Units: seconds.
        taveint: Time interval for accumulation of diagnostic averages.
            For performance purposes, averaging does not have to occur every
            timestep. Units: seconds. Default is 1 (i.e. every 1000 timesteps
            when delta_t = 0.001)
        ntd: Number of threads to use. Should not exceed the number of
            cores on your machine.
        store_as_jax: Store values as jax array instead of numpy array.
            Default is False (store as numpy).
    """
    def __init__(self,
                 beta: float = 0.,
                 rek: float = 0.,
                 rd: float = 0.,
                 H: float = 1.,
                 nx: int = 256,
                 ny: int | None = None,
                 L: float = 2*np.pi,
                 x0: ArrayLike | None = None,
                 delta_t: float = 0.001,
                 taveint: float = 1,
                 ntd: int = 1,
                 time_dim: int | None = None,
                 store_as_jax: bool = False,
                 **kwargs):
        """ Initializes Barotropic object, subclass of Data

        See https://pyqg.readthedocs.io/en/latest/api.html for more details.
        """
        if pyqg is None:
            raise ModuleNotFoundError(
                'No module named \'pyqg\'\n'
                'Barotropic will not work without this optional package\n'
                'To install via conda: conda install -c conda-forge pyqg\n'
                'For more information: '
                'https://pyqg.readthedocs.io/en/latest/installation.html'
                )

        if ny is None:
            ny = nx

        self.m = pyqg.BTModel(beta=beta, rek=rek, rd=rd, H=H, L=L, dt=delta_t,
                              taveint=taveint, ntd=ntd, nx=nx, **kwargs)

        system_dim = self.m.q.size
        super().__init__(system_dim=system_dim, time_dim=time_dim,
                         delta_t=delta_t,
                         store_as_jax=store_as_jax, x0=x0,
                         **kwargs)

    # TODO: Change to produce xarray dataset instead of updating values att.
    def generate(self,
                 n_steps: int | None = None,
                 t_final: float = 40,
                 x0: ArrayLike | None = None
                 ) -> xr.Dataset:
        """Generates values and times, saves them to the data object

        Notes:
            Either provide n_steps or t_final in order to indicate the length
            of the forecast. These are used to set the values, times, and
            time_dim attributes.

        Args:
            n_steps: Number of timesteps. Default is None, which sets
            n_steps to t_final/delta_t
            t_final: Final time of trajectory. Default is 40, which
                results in n_steps = 40000
            x0: the initial conditions. Can also be
                provided when initializing model object. If provided by
                both, the generate() arg takes precedence.
        """

        # Set random seed
        np.random.seed(37)

        # Set n_steps
        if n_steps is None:
            n_steps = int(t_final/self.delta_t)
        else:
            t_final = n_steps * self.delta_t

        # Check if x0 initial conditions is supplied
        # If not, set based on McWilliams 84
        # TODO: Rework so that initial conditions can be supplied in 3D OR 1D
        if x0 is None:
            if self.x0 is not None:
                x0 = self.x0
                if len(x0.shape) != 3:
                    raise ValueError('Initial condition x0 must be a 3D array')
                self.m.set_q(x0)
                self.original_dim = self.x0.shape
                self.x0 = self.x0.flatten()
            else:
                print('Initial condition not set. Start with McWilliams 84 IC '
                      'condition:\n'
                      'doi:10.1017/S0022112084001750')

                fk = self.m.wv != 0
                ckappa = np.zeros_like(self.m.wv2)
                ckappa[fk] = np.sqrt(self.m.wv2[fk]*(1. + (self.m.wv2[fk]/36.)
                                     ** 2)) ** -1

                nhx, nhy = self.m.wv2.shape

                Pi_hat = np.random.randn(nhx, nhy)*ckappa + 1j*np.random.randn(
                    nhx, nhy)*ckappa

                Pi = self.m.ifft(Pi_hat[np.newaxis, :, :])
                Pi = Pi - Pi.mean()
                Pi_hat = self.m.fft(Pi)
                KEaux = self.m.spec_var(self.m.wv * Pi_hat)

                pih = (Pi_hat/np.sqrt(KEaux))
                qih = -self.m.wv2 * pih
                qi = self.m.ifft(qih)
                self.m.set_q(qi)
                self.original_dim = qi.shape
                self.x0 = qi.flatten()
        else:
            self.original_dim = x0.shape
            self.x0 = x0.flatten()

        # Integrate and store values and times
        self.m.dt = self.delta_t
        self.m.tmax = t_final
        self.times = np.arange(0, t_final, self.delta_t)

        # Run simulation
        qs = self.__advance__()

        # Save values
        self.time_dim = qs.shape[0]
        self.values = qs.reshape((self.time_dim, -1))

    # TODO: Remove? I believe this is deprecated
    def forecast(self, n_steps=None, t_final=None, x0=None):
        """Alias for self.generate(), except returns values as output"""
        self.generate(n_steps, t_final, x0)

        return self.values

    def __advance__(self,):
        """Advances the QG model according to set attributes

        Returns:
            Array of absolute potential vorticity (relative potential
            vorticity + background vorticity).
        """
        qs = []
        for _ in self.m.run_with_snapshots(tsnapstart=0, tsnapint=self.m.dt):
            qs.append(deepcopy(self.m.q.squeeze()))
        # Reshape: q was in (ny,nx), qs is now in (nt,nx,ny)
        qs = np.moveaxis(np.array(qs), 1, -1)

        return qs
