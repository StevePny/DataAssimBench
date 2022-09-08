
import logging
import jax.numpy as jnp

from data import Data
from dabench.support.utils import integrate

logging.basicConfig(filename='logfile.log', level=logging.DEBUG)


class DataLorenz63(Data):
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
                 x0=jnp.array([-3.1, -3.1, 20.7]), system_dim=3,
                 input_dim=None, output_dim=None, time_dim=None, values=None,
                 times=None, mean=None, std=None, plot_label=None,
                 noise=0.0, noise_distribution='gaussian',
                 noise_type='multiplicative',
                 **kwargs):
        """Initialize Lorenz63Data object, subclass of Data"""

        # Lorenz63 requires system dim to be 3
        if system_dim is None:
            system_dim = 3
        elif system_dim != 3:
            print('WARNING: input system_dim is {}, setting system_dim = 3.'.format(system_dim))
            system_dim = 3

        super().__init__(system_dim=system_dim, input_dim=input_dim,
                         output_dim=output_dim, time_dim=time_dim,
                         values=values, times=times, delta_t=delta_t,
                         mean=mean, std=std, plot_label=plot_label,
                         noise=noise, noise_distribution=noise_distribution,
                         noise_type=noise_type, **kwargs)

        self.sigma = sigma      # >
        self.rho = rho          # > Model Constants
        self.beta = beta        # >
        self.x0 = x0            # Default initial Conditions

        # create alias
        self.dt = self.delta_t

    def rhs(self, x):
        """vector field of Lorenz 63

        Args:
            x: state vector with shape (system_dim)

        Returns:
            vector field of Lorenz 63 with shape (system_dim)
        """

        dx = jnp.zeros_like(x)
        dx[0] = self.sigma * (x[1] - x[0])
        dx[1] = self.rho * x[0] - x[1] - x[0] * x[2]
        dx[2] = x[0] * x[1] - self.beta*x[2]

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
