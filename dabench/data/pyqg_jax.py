"""Interface with pyqg-jax to run quasigeostrpohic models with autodiff

Requires pyqg-jax: https://pyqg-jax.readthedocs.io/

"""
import logging
import numpy as np
from copy import deepcopy
import jax
import jax.numpy as jnp

from dabench.data import _data

logging.basicConfig(filename='logfile.log', level=logging.DEBUG)

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

    Attributes:
        beta (float): Gradient of coriolis parameter. Units: meters^-1 *
            seconds^-1
        rek (float): Linear drag in lower layer. Units: seconds^-1
        rd (float): Deformation radius. Units: meters.
        delta (float): Layer thickness ratio (H1/H2)
        U1 (float): Upper layer flow. Units: m/s
        U2 (float): Lower layer flow. Units: m/s
        H1 (float): Layer thickness (sets both H1 and H2).
        nx (int): Number of grid points in the x direction.
        ny (int): Number of grid points in the y direction (default: nx).
        L (float): Domain length in x direction. Units: meters.
        W (float): Domain width in y direction. Units: meters (default: L).
        filterfac (float): amplitdue of the spectral spherical filter
            (originally 18.4, later changed to 23.6).
        delta_t (float): Numerical timestep. Units: seconds.
        twrite (int): Interval for cfl writeout. Units: number of timesteps.
        tmax (float): Total time of integration (overwritten by t_final).
            Units: seconds.
        ntd (int): Number of threads to use. Should not exceed the number of
            cores on your machine.
        store_as_jax (bool): Store values as jax array instead of numpy array.
            Default is False (store as numpy).
    """
    def __init__(self,
                 beta=1.5e-11,
                 rd=15000.0,
                 delta=0.25,
                 H1=500,
                 H2=None,
                 U1=0.025,
                 U2=0.0,
                 x0=None,
                 twrite=10000,
                 nx=64,
                 ny=None,
                 delta_t=7200,
                 ntd=1,
                 time_dim=None,
                 values=None,
                 times=None,
                 store_as_jax=False,
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

        self._base_model = pyqg_jax.qg_modelQGModel(
                beta=beta, rd=rd, delta=delta, H1=H1,
                U1=U1, U2=U2, twrite=twrite, ntd=ntd, nx=nx,
                ny=ny, **kwargs)
        self._stepper = pyqg_jax.steppers.AB3Stepper(dt=delta_t)

        self.m = pyqg_jax.steppers.SteppedModel(
                self._base_model, self._stepper
                )
        system_dim = self.m.nx * self.m.ny * self.m.nz
        
        # For pyqg-jax, setting x0 requires a "template" init_state.
        self._template_state = self.m.create_initial_state(
                jax.random.PRNGKey(0)
                )
        super().__init__(system_dim=system_dim, time_dim=time_dim,
                         values=values, times=times, delta_t=delta_t,
                         store_as_jax=store_as_jax, x0=x0,
                         **kwargs)

    @functools.partial(jax.jit, static_argnames=["self", "num_steps"])
    def _roll_out_state(self, state, num_steps):
        """Helper method taken from pyqg-jax docs:
            https://pyqg-jax.readthedocs.io/en/latest/examples.basicstep.html
        """

        def loop_fn(carry, _x):
            current_state = carry
            next_state = stepped_model.step_model(current_state)
            return next_state, next_state

        _final_carry, traj_steps = jax.lax.scan(
            loop_fn, state, None, length=num_steps
        )
        return traj_steps


    def _spec_var(self, ph):
        """Compute variance of p from Fourier coefficients ph

        Note: Taken from original pyqg package:
        https://pyqg.readthedocs.io/en/latest/api.html?highlight=spec_var#pyqg.Model.spec_var
        """

        var_dens = 2. * np.abs(ph)**2 / self.M**2
        # only half of coefs [0] and [nx/2+1] due to symmetry in real fft2
        var_dens[..., 0] = var_dens[...,0]/2.
        var_dens[..., -1] = var_dens[...,-1]/2.

        return var_dens.sum()

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
                fk = self.m.model.wv != 0
                ckappa = np.zeros_like(self.m.model.wv2)
                ckappa[fk] = np.sqrt(
                    self.m.model.wv2[fk]
                    * (1. + (self.m.model.wv2[fk]/36.) ** 2)) ** -1

                nhx, nhy = self.m.model.wv2.shape

                Pi_hat = (np.random.randn(nhx, nhy)*ckappa + 1j *
                          np.random.randn(nhx, nhy)*ckappa)

                Pi = jnp.fft.ifft(Pi_hat[jnp.newaxis, :, :])
                Pi = Pi - Pi.mean()
                Pi_hat = jax.fft.fftfft(Pi)
                KEaux = self._spec_var(self.m.model.wv * Pi_hat)

                pih = (Pi_hat/np.sqrt(KEaux))
                qih = -self.m.model.wv2*pih
                x0 = jax.fft.ifft(qih)

        init_state = self._template_state.update(
                self._template_state.update(
                    state=self._template_state.state.update(
                        q=x0)
                    )
                )

        self.x0 = x0.flatten()

        # Store step times
        self.times = np.arange(0, t_final, self.delta_t)

        # Run simulation
        traj = self._roll_out_state(init_state, num_steps=n_steps)
        qs = traj.state.q

        # Save values
        self.original_dim = qs.shape[1:]
        self.time_dim = qs.shape[0]
        self.values = qs.reshape((self.time_dim, -1))

    def forecast(self, n_steps=None, t_final=None, x0=None):
        """Alias for self.generate(), except returns values as output"""
        self.generate(n_steps, t_final, x0)

        return self.values
