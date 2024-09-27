"""Utils for data generator integration"""

import logging
import numpy as np
import jax.numpy as jnp
from scipy.integrate import odeint as spodeint
from jax.experimental.ode import odeint

logging.basicConfig(filename='logfile.log', level=logging.DEBUG)


def integrate(function, x0, t_final, delta_t, method='odeint', stride=None,
              jax_comps=True,
              **kwargs):
    """ Integrate forward in time.

    Args:
        function (ndarray): the model equations to integrate
        x0 (ndarray): initial conditions state vector with shape (system_dim)
        t_final (float): the final absolute time
        delta_t (float): timestep size
        method (str): Integration method, one of 'odeint', 'euler',
            'adambash2', 'ode_adambash2', 'rk2'. Right now, only odeint is
            implemented
        stride (float): stride for output data
        **kwargs: keyword arguments for the integrator

    Returns:
        Tuple of (y, t) where y is ndarray of state at each timestep with shape
        (time_dim, system_dim) and t is time array with shape (time_dim)
    """
    if method == 'odeint':
        # Define timesteps
        t = np.arange(0.0, t_final - delta_t/2, delta_t)
        # If stride is defined, remove timesteps that are not on stride steps
        if stride is not None:
            assert stride > 1 and isinstance(stride, int), \
                'integrate: stride = {}, must be > 1 and an int'.format(stride)
            t = t[::stride]
        if jax_comps:
            y = odeint(function, x0, t, **kwargs)
        else:
            y = spodeint(function, x0, t, **kwargs)
    else:
        raise 'integration method {} not supported'.format(method)

    return y, t
