"""Interface with qgs to run quasigeostrpohic models

Requires qgs: https://qgs.readthedocs.io/

"""
import logging
import numpy as np
from copy import deepcopy
import xarray as xr
import jax
import jax.numpy as jnp
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

# For typing
ArrayLike = np.ndarray | jax.Array


class QGS(_data.Data):
    """QGS quasi-geostrophic model data generator.

    The QGS class is simply a wrapper of an *optional* qgs package.
    See https://qgs.readthedocs.io/

    Args:
        model_params: qgs parameter object. See:
            https://qgs.readthedocs.io/en/latest/files/technical/configuration.html#qgs.params.params.QgParams
            If None, will use defaults specified by:
            De Cruz, et al. (2016). Geosci. Model Dev., 9, 2793-2808.
        x0: Initial state vector, array of floats. Default is:
        delta_t: Numerical timestep. Units: seconds.
        store_as_jax: Store values as jax array instead of numpy array.
            Default is False (store as numpy).
    """
    def __init__(self,
                 model_params: QgParams | None = None,
                 x0: ArrayLike | None = None,
                 delta_t: ArrayLike | None =  0.1,
                 system_dim: int | None = None,
                 store_as_jax: bool = False,
                 random_seed: int = 37,
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
        else:
            self.model_params = model_params
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

        super().__init__(system_dim=system_dim,
                         delta_t=delta_t, store_as_jax=store_as_jax, x0=x0,
                         **kwargs)

        self.f, self.Df = create_tendencies(self.model_params)

    def _create_default_qgparams(self) -> QgParams:
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

    def rhs(self,
            x: ArrayLike,
            t: float | None = 0
            ) -> np.ndarray:
        """Vector field (tendencies) of qgs system

        Args:
            x: State vector, shape: (system_dim)
            t: times vector. Required as argument slot for some numerical
                integrators but unused.
        Returns:
            Vector field of qgs

        """

        dx = self.f(t, x)

        return dx

    def Jacobian(self,
                 x: ArrayLike,
                 t: float | None = 0
                 ) -> np.ndarray:
        """Jacobian of the qgs system

        Args:
            x: State vector, shape: (system_dim)
            t: times vector. Required as argument slot for some numerical
                integrators but unused.

        Returns:
            J: Jacobian matrix, shape: (system_dim, system_dim)

        """

        J = self.Df(t, x)

        return J

    def generate(self,
                 n_steps: int | None = None,
                 t_final: float | None = None,
                 x0: ArrayLike | None = None,
                 M0: ArrayLike | None = None,
                 return_tlm: bool = False,
                 stride: int | None = None,
                 **kwargs) -> xr.Dataset | tuple[xr.Dataset | xr.DataArray]:
        """Generates a dataset and assigns values and times to the data object.

        Notes:
            Either provide n_steps or t_final in order to indicate the length
            of the forecast.

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
            Xarray Dataset of output vector and (if return_tlm=True)
                Xarray DataArray of TLMs corresponding to the system trajectory.
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

        # Convert to JAX if necessary
        time_dim = t.shape[0]
        out_dim = (time_dim,) + self.original_dim
        if self.store_as_jax:
            y_out = jnp.array(y[:,:self.system_dim].reshape(out_dim))
        else:
            y_out = np.array(y[:,:self.system_dim].reshape(out_dim))
        # Build Xarray object for output
        coord_dict = dict(zip(
            ['time'] + self.coord_names,
            [t] + [np.arange(dim) for dim in self.original_dim]
        ))
        out_vec = xr.Dataset(
            {self.var_names[0]: (coord_dict.keys(),y_out)},
            coords=coord_dict,
            attrs={'store_as_jax':self.store_as_jax,
                   'system_dim': self.system_dim,
                   'delta_t': self.delta_t
            }
        )

        # Return the data series and associated TLMs if requested
        if return_tlm:
            # Reshape M matrix
            if self.store_as_jax:
                M = jnp.reshape(y[:, self.system_dim:],
                                (time_dim,
                                self.system_dim,
                                self.system_dim)
                                )
            else:
                M = np.reshape(y[:, self.system_dim:],
                                (time_dim,
                                self.system_dim,
                                self.system_dim)
                                )
            M = xr.DataArray(
                M, dims=('time','system_0','system_n')
            )
            return out_vec, M
        else:
            return out_vec

    def rhs_aux(self,
                x: ArrayLike,
                t: ArrayLike
                ) -> jax.Array:
        """The auxiliary model used to compute the TLM.

        Args:
          x: State vector with size (system_dim)
          t: Array of times with size (time_dim)

        Returns:
            State vector [size: (system_dim,)]
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

    def calc_lyapunov_exponents_series(
            self,
            total_time: float | None =  None,
            rescale_time: float = 1,
            convergence: float = 0.01,
            x0: ArrayLike | None = None
            ) -> ArrayLike:
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
            total_time: Time to integrate over to compute LEs.
                Usually there's a tradeoff between accuracy and computation
                time (more total_time leads to higher accuracy but more
                computation time). Default depends on model type and are based
                roughly on how long it takes for satisfactory convergence:
                For Lorenz63: n_steps=15000 (total_time=150 for delta_t=0.01)
                For Lorenz96: n_steps=50000 (total_time=500 for delta_t=0.01)
            rescale_time: Time for when the algorithm rescales the
                propagator to reduce the exponential growth in errors.
                Default is 1 (i.e. 100 timesteps when delta_t = 0.01).
            convergence: Prints warning if LE convergence is below
                this number. Default is 0.01.
            x0: initial condition to start computing LE.  Needs
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

            traj, M = self.generate(t_final=t2-t1, x0=x0, M0=M0, return_tlm=True)
            x_t2 = traj.isel(time=-1).to_array().data.flatten()
            M_t2 = M.isel(time=-1).data

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
