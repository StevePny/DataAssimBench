"""Deterministic metrics"""

import jax.numpy as jnp
import numpy as np
import jax

from dabench.metrics import _utils

# For typing
ArrayLike = np.ndarray | jax.Array


__all__ = [
    'pearson_r',
    'rmse',
    'mse',
    'mae'
    ]


def pearson_r(
        predictions: ArrayLike,
        targets: ArrayLike
        ) -> jax.Array:
    """JAX array implementation of Pearson R

    Args:
        predictions (ndarray): Array of predictions
        targets (ndarray): Array of targets to compare against. Shape must
            be broadcastable to shape of predictions.

    Returns:
        Float, Pearson's R correlation coefficient.
    """
    targ_devs = targets - jnp.mean(targets)
    pred_devs = predictions - jnp.mean(predictions)
    top = jnp.sum(targ_devs*pred_devs)
    bottom = (jnp.sqrt(jnp.sum(jnp.square(targ_devs)))
              * jnp.sqrt(jnp.sum(jnp.square(pred_devs))))

    return top/bottom

def mse(
        predictions: ArrayLike,
        targets: ArrayLike
        ) -> jax.Array:
    """JAX array implementation of Mean Squared Error

    Args:
        predictions (ndarray): Array of predictions
        targets (ndarray): Array of targets to compare against. Shape must
            be broadcastable to shape of predictions.

    Returns:
        Float, Mean Squared Error
    """
    return jnp.mean(jnp.square(predictions - targets))

def rmse(
         predictions: ArrayLike,
         targets: ArrayLike
         ) -> jax.Array:
    """JAX array implementation of Root Mean Squared Error

    Args:
        predictions (ndarray): Array of predictions
        targets (ndarray): Array of targets to compare against. Shape must
            be broadcastable to shape of predictions.

    Returns:
        Float, Root Mean Squared Error
    """
    return jnp.sqrt(mse(predictions, targets))

def mae(
        predictions: ArrayLike,
        targets: ArrayLike
        ) -> jax.Array:
    """JAX array implementation of Mean Absolute Error

    Args:
        predictions (ndarray): Array of predictions
        targets (ndarray): Array of targets to compare against. Shape must
            be broadcastable to shape of predictions.

    Returns:
        Float, Mean Absolute Error
    """
    return jnp.mean(jnp.abs(predictions - targets))
