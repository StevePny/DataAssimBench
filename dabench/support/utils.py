"""Utils for integration"""

import logging
import jax.numpy as jnp
from jax.experimental.ode import odeint

logging.basicConfig(filename='logfile.log', level=logging.DEBUG)


def integrate(function, x0, t_final, delta_t, method='odeint', stride=None,
              **kwargs):
    """ Integrate forward in time.

    Args:
        function (ndarray): the model equations to integrate
        x0 (ndarray): initial conditions state vector
        t_final (float): the final absolute time
        delta_t (float): timestep size
        method (str): Integration method, one of 'odeint', 'euler',
            'adambash2', 'ode_adambash2', 'rk2'. Right now, only odeint is
            implemented
        stride (float): stride for output data
        **kwargs: keyword arguments for the integrator

    Returns:
        Tuple of (y, t) where y is ndarray of state at each timestep and t is
        time array
    """

    if method == 'odeint':
        t = jnp.arange(0.0, t_final, delta_t)
        if stride is not None:
            assert stride > 1 and isinstance(stride, int), \
                'integrate: stride = {}, must be > 1 and an int'.format(stride)
            t = t[::stride]
        y = odeint(function, x0, t, **kwargs).transpose()
    else:
        raise 'integration method {} not supported'.format(method)

    return y.T, t
