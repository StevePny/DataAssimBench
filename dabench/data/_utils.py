"""Utils for data generator integration"""

import logging
import numpy as np
import jax
import jax.numpy as jnp
from scipy.integrate import odeint as spodeint
from jax.experimental.ode import odeint


# For typing
ArrayLike = np.ndarray | jax.Array

logging.basicConfig(filename='logfile.log', level=logging.DEBUG)

def integrate(function: ArrayLike,
              x0: ArrayLike,
              t_final: float,
              delta_t: float,
              method: str = 'odeint',
              stride: float | None = None,
              jax_comps: bool = True,
              **kwargs
              ) -> tuple[ArrayLike, ArrayLike]:
    """Integrate forward in time.

    Args:
        function: the model equations to integrate
        x0: initial conditions state vector with shape (system_dim)
        t_final: the final absolute time
        delta_t (float): timestep size
        method: Integration method, one of 'odeint', 'euler',
            'adambash2', 'ode_adambash2', 'rk2'. Right now, only odeint is
            implemented
        stride: stride for output data
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
