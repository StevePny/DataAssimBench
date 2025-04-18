"""Interface with pyqg-jax to run quasigeostrpohic models with autodiff

Requires pyqg-jax: https://pyqg-jax.readthedocs.io/

"""
import logging
import numpy as np
from copy import deepcopy
import functools
import jax
import jax.numpy as jnp
import xarray as xr
import jax.numpy as jnp

from dabench.data import _data

logging.basicConfig(filename='logfile.log', level=logging.DEBUG)

# For typing
ArrayLike = np.ndarray | jax.Array

try:
    import pyqg_jax
except ImportError:
    pyqg = None
    logging.warning(
        'Package: pyqg-jax not found!\n'
        'PyQGJax will not work without this optional package\n'
        'To install via pip: python -m pip install pyqg-jax\n'
        'For more information: https://pyqg-jax.readthedocs.io/en/latest/install.html '
        )


class PyQGJax(_data.Data):
    """Class to set up quasi-geotropic model

    The PyQGJax class is simply a wrapper of the "optional" pyqg-jax package.
    See https://pyqg-jax.readthedocs.io

    Notes:
        Uses default attribute values from pyqg_jax.QGModel:
        https://pyqg.readthedocs.io/en/latest/api.html#pyqg.QGModel

    Args:
        beta: Gradient of coriolis parameter. Units: meters^-1 *
            seconds^-1
        rd: Deformation radius. Units: meters.
        delta: Layer thickness ratio (H1/H2)
        H1: Layer thickness (sets both H1 and H2 if H2 not specified).
        H2: Layer 2 thickness.
        U1: Upper layer flow. Units: m/s
        U2: Lower layer flow. Units: m/s
        x0: the initial conditions. Can also be
            provided when initializing model object. If provided by
            both, the generate() arg takes precedence.
        nx: Number of grid points in the x direction.
        ny: Number of grid points in the y direction (default: nx).
        delta_t: Numerical timestep. Units: seconds.
        store_as_jax: Store values as jax array instead of numpy array.
            Default is False (store as numpy).
    """
    def __init__(self,
                 beta: float = 1.5e-11,
                 rd: float = 15000.0,
                 delta: float = 0.25,
                 H1: float = 500,
                 H2: float | None = None,
                 U1: float = 0.025,
                 U2: float = 0.0,
                 x0: ArrayLike | None = None,
                 nx: int = 64,
                 ny: int | None = None,
                 delta_t: float = 7200,
                 random_seed: int = 37,
                 time_dim: int | None = None,
                 store_as_jax: bool = False,
                 **kwargs):
        """ Initialize PyQGJax QGModel object, subclass of Base

        See https://pyqg-jax.readthedocs.io/en/latest/api.html for details.
        """

        if pyqg_jax is None:
            raise ModuleNotFoundError(
                'No module named \'pyqg_jax\'\n'
                'PyQGJax will not work without this optional package\n'
                'To install via conda: conda install -c conda-forge pyqg_jax\n'
                'For more information: '
                'https://pyqg.readthedocs.io/en/latest/installation.html'
                )

        self.random_seed = random_seed
        self._rng = np.random.default_rng(self.random_seed)

        self._base_model = pyqg_jax.qg_model.QGModel(
                beta=beta, rd=rd, delta=delta, H1=H1,
                U1=U1, U2=U2, nx=nx, ny=ny,
                precision=pyqg_jax.state.Precision.DOUBLE, **kwargs)

        self._stepper = pyqg_jax.steppers.AB3Stepper(dt=delta_t)

        self.m = pyqg_jax.steppers.SteppedModel(
                self._base_model, self._stepper
                )
        system_dim = self._base_model.nx * self._base_model.ny * self._base_model.nz
        original_dim = (self._base_model.nz, self._base_model.nx, self._base_model.ny)

        # For pyqg-jax, setting x0 requires a "template" init_state.
        self._template_state = self.m.create_initial_state(
                jax.random.PRNGKey(0)
                )
        super().__init__(system_dim=system_dim, original_dim=original_dim,
                         time_dim=time_dim, delta_t=delta_t,
                         store_as_jax=store_as_jax, x0=x0,
                         **kwargs)

    @functools.partial(jax.jit, static_argnames=["self", "num_steps"])
    def _roll_out_state(self, state, num_steps):
        """Helper method taken from pyqg-jax docs:
            https://pyqg-jax.readthedocs.io/en/latest/examples.basicstep.html
        """

        def loop_fn(carry, _x):
            current_state = carry
            next_state = self.m.step_model(current_state)
            return next_state, next_state

        _final_carry, traj_steps = jax.lax.scan(
            loop_fn, state, None, length=num_steps
        )
        return traj_steps


    def _spec_var(self,
                  ph: np.ndarray
                    ) -> float:
        """Compute variance of p from Fourier coefficients ph

        Note: Taken from original pyqg package:
        https://pyqg.readthedocs.io/en/latest/api.html?highlight=spec_var#pyqg.Model.spec_var
        """

        var_dens = 2. * np.abs(ph)**2 / self._base_model.M**2
        # only half of coefs [0] and [nx/2+1] due to symmetry in real fft2
        var_dens[..., 0] = var_dens[...,0]/2.
        var_dens[..., -1] = var_dens[...,-1]/2.

        return var_dens.sum()

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
            n_steps: Number of timesteps. One of n_steps OR
                t_final must be specified.
            t_final: Final time of trajectory. One of n_steps OR
                t_final must be specified.
            x0: the initial conditions. Can also be
                provided when initializing model object. If provided by
                both, the generate() arg takes precedence.
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
        # TODO: Rework so that x0 can be supplied in 1, 2, or 3D
        if x0 is None:
            if self.x0 is not None:
                x0 = self.x0
                if (len(x0.shape) != 3) and (x0.shape[0] != 2):
                    raise ValueError(
                        'Initial condition x0 must be 3D array and the first '
                        'dimension must be for this 2-layer QG model')
            else:
                print('Initial condition not set. Start with random IC.')

                fk = self._base_model.wv != 0
                ckappa = np.zeros_like(self._base_model.wv2)
                ckappa[fk] = np.sqrt(
                    self._base_model.wv2[fk]
                    * (1. + (self._base_model.wv2[fk]/36.) ** 2)) ** -1

                nhx, nhy = self._base_model.wv2.shape

                Pi_hat = (self._rng.standard_normal((nhx, nhy))*ckappa + 1j *
                          self._rng.standard_normal((nhx, nhy))*ckappa)

                Pi = np.repeat(jnp.fft.irfft2(Pi_hat[np.newaxis,:,:]),self._base_model.nz,  axis=0)
                Pi = Pi - Pi.mean()
                Pi_hat = jnp.fft.rfft2(Pi)
                KEaux = self._spec_var(self._base_model.wv * Pi_hat)
                pih = (Pi_hat/np.sqrt(KEaux))
                qih = -self._base_model.wv2*pih
                x0 = jnp.fft.irfft2(qih)

        init_state = self._template_state.update(
                state=self._template_state.state.update(
                    q=x0
                    )
                )

        self.x0 = x0.flatten()

        # Store step times
        self.times = jnp.arange(0, t_final, self.delta_t)

        # Run simulation
        traj = self._roll_out_state(init_state, num_steps=n_steps)
        qs = traj.state.q

        # Save values
        self.time_dim = qs.shape[0]
        self.values = qs.reshape((self.time_dim, -1))

    # TODO: Remove? Believe this is deprecated
    def forecast(self, n_steps=None, t_final=None, x0=None):
        """Alias for self.generate(), except returns values as output"""
        self.generate(n_steps, t_final, x0)

        return self.values
