"""Lorenz 1963 3-variable model data generation"""

import logging
import numpy as np
import jax
import jax.numpy as jnp

from dabench.data import _data

logging.basicConfig(filename='logfile.log', level=logging.DEBUG)

# For typing
ArrayLike = np.ndarray | jax.Array

class Lorenz63(_data.Data):
    """ Class to set up Lorenz 63 model data

    Attributes:
        sigma: Lorenz 63 param. Default is 10., the original value
            used in Lorenz, 1963.
            https://doi.org/10.1175/1520-0469(1963)020<0130:DNF>2.0.CO;2
        rho: Lorenz 63 param. Default is 28., the value used in
            Lorenz, 1963 (see DOI above)
        beta: Lorenz 63 param. Default is 8./3., the value used in
            Lorenz, 1963 (see DOI above)
        delta_t: length of one time step
        x0: Initial state, array of floats of size
            (system_dim). Default is jnp.array([-10.0, -15.0, 21.3]), which
            is the system state after a 6000 step spinup with delta_t=0.01
            and initial conditions [0., 1., 0.], a spinup which replicates
            the simulation described in Lorenz, 1963.
        system_dim: system dimension. Must be 3 for Lorenz63.
        time_dim: total time steps
        store_as_jax: Store values as jax array instead of numpy array.
            Default is False (store as numpy).
    """

    def __init__(self,
                 sigma: float = 10.,
                 rho: float = 28.,
                 beta: float = 8./3.,
                 delta_t: float = 0.01,
                 x0: ArrayLike | None = jnp.array([-10.0, -15.0, 21.3]),
                 system_dim: int = 3,
                 time_dim: int | None = None,
                 values: ArrayLike | None = None,
                 store_as_jax: bool = False,
                 **kwargs):
        """Initialize Lorenz63 object, subclass of Base"""

        # Lorenz63 requires system dim to be 3
        if system_dim is None:
            system_dim = 3
        elif system_dim != 3:
            print('WARNING: input system_dim is {}, '
                  'Lorenz63 requires system_dim=3.'.format(system_dim))
            print('Assigning system_dim to 3.')
            system_dim = 3

        super().__init__(system_dim=system_dim, time_dim=time_dim,
                         values=values, delta_t=delta_t,
                         store_as_jax=store_as_jax, **kwargs)

        # Model constants
        self.sigma = sigma
        self.rho = rho  # Model Constants
        self.beta = beta

        # Initial conditions
        self.x0 = x0

    def rhs(self,
            x: ArrayLike,
            t: ArrayLike | None =  None
            ) -> jax.Array:
        """vector field of Lorenz 63

        Args:
            x: state vector with shape (system_dim)
            t: times vector. Needed as argument slot for odeint but unused

        Returns:
            vector field of Lorenz 63 with shape (system_dim)
        """

        dx = jnp.array([
            self.sigma * (x[1] - x[0]),
            self.rho * x[0] - x[1] - x[0] * x[2],
            x[0] * x[1] - self.beta*x[2]
            ])

        return dx

    def Jacobian(self,
                 x: ArrayLike
                 ) -> jax.Array:
        """Jacobian of the L63 system

        Args:
            x: state vector with shape (system_dim)

        Returns:
            ndarray of Jacobian matrix with shape (system_dim, system_dim)
        """

        s = self.sigma
        b = self.beta
        r = self.rho

        J = jnp.array([[    -s,     s,     0],
                       [r-x[2],    -1, -x[0]],
                       [  x[1],  x[0],    -b]])

        return J

