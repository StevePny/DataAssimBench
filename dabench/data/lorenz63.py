
import logging
import jax.numpy as jnp

from dabench.data import data
from dabench.support.utils import integrate

logging.basicConfig(filename='logfile.log', level=logging.DEBUG)


class DataLorenz63(data.Data):
    """ Class to set up Lorenz 63 model data

    Attributes:
        system_dim (int): system dimension
        time_dim (int): total time steps
        sigma (float): Lorenz 63 params
        rho (float): Lorenz 63 params
        beta (float): Lorenz 63 params
        delta_t (float): length of one time step
    """

    def __init__(self, sigma=10., rho=28., beta=8./3., delta_t=0.01,
                 x0=jnp.array([-3.1, -3.1, 20.7]), system_dim=3, time_dim=None,
                 values=None, **kwargs):
        """Initialize Lorenz63Data object, subclass of Data"""

        # Lorenz63 requires system dim to be 3
        if system_dim is None:
            system_dim = 3
        elif system_dim != 3:
            print('WARNING: input system_dim is {}, setting system_dim = 3.'.format(system_dim))
            print('Assigning system_dim to 3.')
            system_dim = 3

        super().__init__(system_dim=system_dim, time_dim=time_dim,
                         values=values, delta_t=delta_t, **kwargs)

        self.sigma = sigma
        self.rho = rho  # Model Constants
        self.beta = beta
        self.x0 = x0  # Default initial Conditions

        # create alias
        self.dt = self.delta_t

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

