"""Lorenz 96 model data generator"""
import logging
import jax.numpy as jnp

from dabench.data import data

logging.basicConfig(filename='logfile.log', level=logging.DEBUG)


class DataLorenz96(data.Data):
    """Class to set up Lorenz 96 model data.

    Notes:
        Default values come from Lorenz, 1996:
        eapsweb.mit.edu/sites/default/files/Predicability_a_Problem_2006.pdf

    Attributes:
        forcing_term (float): Forcing constant for Lorenz96, prevents energy
            from decaying to 0. Default is 8.0.
        x0 (ndarray, float): Initial state vector, array of floats of size
            (system_dim). For system_dim of 5, 6, or 36, defaults are the final
            state of a 14400 timestep spinup with delta_t=0.05 starting with an
            initial state (x0) of all 0s except the first element, which is set
            to 0.01. This setup is taken from Lorenz, 1996. For all other
            system_dim settings, default is all 0s except the first element,
            which is set to 0.01.
        system_dim (int): System dimension, must be between 5 and 40.
            Default is 36.
        time_dim (int): Total time steps
        delta_t (float): Length of one time step. Defaults is 0.05.
            Lorenz, 1996.
    """

    def __init__(self,
                 forcing_term=8.,
                 delta_t=0.05,
                 x0=None,
                 system_dim=36,
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

        # Defaults come from 14400 timestep spinup with delta_t=0.05
        initial_defaults_dict = {
            5: jnp.array([3.552747, 5.173696, 5.4161925, 5.1463985,
                          -0.62878036]),
            6: jnp.array([-1.277197, 0.5191239, 2.1629903, 4.641832, 5.013882,
                          -3.1474972]),
            36: jnp.array([0.90061724,  2.2108543,  3.3563306,  7.0470520,
                           7.3828993, -2.2906365,  1.6358340,  4.5246205,
                           -0.8536633,  2.2018400,  2.5094680,  5.6148005,
                           -1.7163916, -3.5827417,  0.22293478,  1.8138107,
                           3.7354333,  5.9006715, -4.6722836,  0.4664867,
                           0.36800075,  7.7004447,  3.0569422, -1.7238870,
                           -2.1296368,  1.6388168,  5.1955190,  4.7863874,
                           0.8382774, -4.0938597,  0.5181451,  1.2503184,
                           6.0076460,  7.1161866, -3.2190716, -2.3532054])
            }
        # Set initial conditions
        if x0 is None:
            if system_dim in initial_defaults_dict.keys():
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