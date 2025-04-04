"""Helper functions for metrics"""

import jax.numpy as jnp
import numpy as np
import jax

# For typing
ArrayLike = np.ndarray | jax.Array


def _cov(
        a: ArrayLike,
        b: ArrayLike
        ) -> jax.Array:
    """Covariance"""
    a_mean = jnp.mean(a)
    b_mean = jnp.mean(b)

    return ((a-a_mean)*(b - b_mean)).sum()/(a.shape[0]-1)

    