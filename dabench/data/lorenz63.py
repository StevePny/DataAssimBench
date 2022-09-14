
import logging
import jax.numpy as jnp

from dabench.data import data
from dabench.support.utils import integrate

logging.basicConfig(filename='logfile.log', level=logging.DEBUG)


class DataLorenz63(data.Data):
    """ Class to set up Lorenz 63 model data

    Attributes:
        sigma (float): Lorenz 63 param. Default is 10., the original value
            used in Lorenz, 1963.
            https://doi.org/10.1175/1520-0469(1963)020<0130:DNF>2.0.CO;2
        rho (float): Lorenz 63 param. Default is 28., the value used in
            Lorenz, 1963 (see DOI above)
        beta (float): Lorenz 63 param. Default is 8./3., the value used in
            Lorenz, 1963 (see DOI above)
        x0 (ndarray, float): Initial state, array of floats of size
            (system_dim). Default is jnp.array([-7.5, -11.5, 18.5]), which
            is the system state after a 14000 step spinup with delta_t=0.01
            and initial conditions [1., 1., -1.]
        system_dim (int): system dimension. Must be 3 for DataLorenz63.
        time_dim (int): total time steps
        delta_t (float): length of one time step
    """

    def __init__(self,
                 sigma=10.,
                 rho=28.,
                 beta=8./3.,
                 delta_t=0.01,
                 x0=jnp.array([-7.5, -11.5, 18.5]),
                 system_dim=3,
                 time_dim=None,
                 values=None,
                 **kwargs):
        """Initialize Lorenz63Data object, subclass of Data"""

        # Lorenz63 requires system dim to be 3
        if system_dim is None:
            system_dim = 3
        elif system_dim != 3:
            print('WARNING: input system_dim is {}, '
                  'DataLorenz63 requires system_dim=3.'.format(system_dim))
            print('Assigning system_dim to 3.')
            system_dim = 3

        super().__init__(system_dim=system_dim, time_dim=time_dim,
                         values=values, delta_t=delta_t, **kwargs)

        # Model constants
        self.sigma = sigma
        self.rho = rho  # Model Constants
        self.beta = beta

        # Initial conditions
        self.x0 = x0

    def rhs(self, x, t=None):
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

    def Jacobian(self, x):
        """ Jacobian of the L63 system

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

