"""Helper functions for metrics"""

import jax.numpy as jnp


def _cov(a, b):
    """Covariance"""
    a_mean = jnp.mean(a)
    b_mean = jnp.mean(b)

    return ((a-a_mean)*(b - b_mean)).sum()/(a.shape[0]-1)

    