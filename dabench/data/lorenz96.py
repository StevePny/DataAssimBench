"""Lorenz 96 model"""
import logging
import jax.numpy as jnp

from dabench.data import data
from dabench.support.utils import integrate

logging.basicConfig(filename='logfile.log', level=logging.DEBUG)


class DataLorenz96(data.Data):
    """ Class to set up Lorenz 96 model data

    Attributes:
        forcing_term (float): Forcing constant for Lorenz96, prevents energy
            from decaying to 0. Default is 8, from Lorenz, 1998.
            https://doi.org/10.1175/1520-0469(1998)055<0399:OSFSWO>2.0.CO;2
        x0 (ndarray, float): Initial state vector, array of floats of size
            (system_dim). For system_dim of 5 or 6, defaults are set from a
            20000 timestep spinup with delta_t=0.01. For system_dim > 6,
            default is all 0s except the first element, which is set to 0.01.
        system_dim (int): System dimension, must be between 5 and 40.
            Default is 5.
        time_dim (int): Total time steps
        delta_t (float): Length of one time step, default is 0.01.
    """

    def __init__(self,
                 forcing_term=8.,
                 delta_t=0.01,
                 x0=None,
                 system_dim=3,
                 time_dim=None,
                 values=None,
                 **kwargs):
        """Initialize DataLorenz96 object, subclass of Data"""

        super().__init__(system_dim=system_dim, time_dim=time_dim,
                         values=values, delta_t=delta_t, **kwargs)

        # Check system_dim >= 5
        if system_dim < 5:
            raise ValueError(('system_dim is {}, for Lorenz96 it must '
                              'be >5'.format(self.system_dim)))

        # Model constants
        self.forcing_term = forcing_term
        self.delta_t = delta_t

        # Set initial conditions
        if x0 is None:
            if system_dim <= 6:
                # Defaults come from 20000 timestep spinup with delta_t=0.01
                initial_defaults_dict = {
                    5: jnp.array([7.97355787, 7.97897913, 8.00370696, 
                                  7.98444298, 7.97446945]),
                    6: jnp.array([1.50995768, 1.56066941, 4.25604607,
                                  7.25271592, -3.11392061, 2.26497509])
                    }
                x0 = initial_defaults_dict[system_dim]
            else:
                # Set to all 0s except first element = 0.01
                x0 = jnp.concatenate([jnp.array([0.01]),
                                      jnp.zeros(system_dim-1)])
        self.x0 = x0

    def rhs(self, x, t=None):
        """Computes vector field of Lorenz 96

        Args:
            x: state vector with shape (system_dim)
            t: times vector. Required as argument slot for some numerical
                integrators but unused

        Returns:
            vector field of Lorenz 96 with shape (system_dim)
        """

        # Shortnames for constants for simplicity
        N = self.system_dim
        F = self.forcing_term

        dx = jnp.concatenate([
            # Edge cases, i=1 and i=2
            jnp.array([(x[1] - x[N-2]) * x[N-1] - x[0],
                       (x[2] - x[N-1]) * x[0] - x[1]]),
            # General case for i=3 to N
            (x[3:N] - x[:N-3]) * x[1:N-2] - x[2:N-1],
            # Edge case i=N
            jnp.array([(x[0] - x[N - 3]) * x[N - 2] - x[N - 1]])
                ])

        # Add forcing term
        dx += F

        return dx

    def Jacobian(self, x):
        """Computes the Jacobian of the Lorenz96 system

        Args:
            x: state vector with shape (system_dim)

        Returns:
            ndarray of Jacobian matrix with shape (system_dim, system_dim)
        """
        # -1.0 on the diagonal
        J = -jnp.identity(self.system_dim)

        # Filling in values on off-diagonals
        i = jnp.arange(self.system_dim)
        jm2 = jnp.mod(i-2, self.system_dim)
        jm1 = jnp.mod(i-1, self.system_dim)
        jp1 = jnp.mod(i+1, self.system_dim)
        J = J.at[(i, jm2)].set(-x[jm1])
        J = J.at[(i, jm1)].set(x[jp1] - x[jm2])
        J = J.at[(i, jp1)].set(x[jm1])

        return J
