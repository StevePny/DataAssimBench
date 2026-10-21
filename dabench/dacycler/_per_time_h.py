"""Memory-light per-time linear observation operator.

The 4D cyclers index a per-obs-time operator ``Hs[i]`` of shape
``(obs_dim, system_dim)``.  Materialising ``Hs`` as a dense
``(n_times, obs_dim, system_dim)`` array (``jnp.repeat(H, T)`` or
``mask[:, :, None] * H``) costs ``T`` copies of ``H`` and OOMs for large
observation networks.  :class:`PerTimeH` keeps a single ``H`` of shape
``(obs_dim, system_dim)`` plus an optional ``(n_times, obs_dim)`` mask, and
applies the mask to the PRODUCT:

    (mask_t[:, None] * H) @ x == mask_t[:, None] * (H @ x)

Adjoints (``jax.grad`` / ``jax.vjp``) follow automatically:
``H_t^T r == H^T (mask_t * r)``.  Masked-out rows contribute exactly zero.
"""

from typing import Any

import jax
import jax.numpy as jnp


class _PerTimeHRow:
    """``H_t`` for one obs time: ``H`` with rows scaled by ``mask_t``."""

    def __init__(self, H: jax.Array, mask_t: jax.Array | None):
        self.H = H
        self.mask_t = mask_t
        self.shape = H.shape
        self.dtype = H.dtype

    def __matmul__(self, x: Any) -> jax.Array:
        out = self.H @ x
        if self.mask_t is None:
            return out
        m = self.mask_t.astype(out.dtype)
        if out.ndim == 2:
            m = m[:, None]
        return m * out


@jax.tree_util.register_pytree_node_class
class PerTimeH:
    """Lazy ``(n_times, obs_dim, system_dim)`` stack of masked copies of H.

    Args:
        H: Single-time operator ``(obs_dim, system_dim)``.
        n_times: Number of obs times in the window.
        mask: Optional ``(n_times, obs_dim)`` active mask (bool or 0/1).
            ``None`` means every row is active (plain ``repeat``).
    """

    def __init__(self, H: Any, n_times: int, mask: Any | None = None):
        self.H = jnp.asarray(H)
        self.n_times = int(n_times)
        self.mask = None if mask is None else jnp.asarray(mask)

    @property
    def shape(self) -> tuple[int, int, int]:
        return (self.n_times,) + tuple(self.H.shape)

    @property
    def dtype(self):
        return self.H.dtype

    def __getitem__(self, i: Any) -> _PerTimeHRow:
        mask_t = None if self.mask is None else self.mask[i]
        return _PerTimeHRow(self.H, mask_t)

    def dense(self) -> jax.Array:
        """Materialise ``(n_times, obs_dim, system_dim)`` (small problems /
        dense diagnostics only)."""
        Hs = jnp.broadcast_to(self.H[None], self.shape)
        if self.mask is None:
            return Hs
        return self.mask[:, :, None] * Hs

    def astype(self, dtype: Any) -> "PerTimeH":
        return PerTimeH(self.H.astype(dtype), self.n_times, self.mask)

    def tree_flatten(self):
        return (self.H, self.mask), self.n_times

    @classmethod
    def tree_unflatten(cls, n_times, children):
        obj = object.__new__(cls)
        obj.H, obj.mask = children
        obj.n_times = n_times
        return obj


def make_per_time_H(H: Any, n_times: int, obs_loc_mask: Any,
                    stationary: Any) -> PerTimeH:
    """``PerTimeH`` for a user-supplied single-time ``H``.

    Stationary networks skip the mask (it is all-ones there anyway).
    """
    is_stat = isinstance(stationary, bool) or hasattr(stationary, "item")
    if is_stat:
        try:
            is_stat = bool(stationary)
        except Exception:  # traced value: fall back to masking
            is_stat = False
    return PerTimeH(H, n_times, None if is_stat else obs_loc_mask)


def as_dtype(Hs: Any, dtype: Any) -> Any:
    if isinstance(Hs, PerTimeH):
        return Hs.astype(dtype)
    return jnp.asarray(Hs, dtype)
