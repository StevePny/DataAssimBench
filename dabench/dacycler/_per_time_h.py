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

``H`` may also be a :class:`LinearObsOperator`: a matrix-free linear map
``x (system_dim,) -> (obs_dim,)`` given as a JAX function plus shape
metadata.  It supports ``H @ x`` (1-D state or ``(system_dim, k)`` column
batch, vmapped) and ``H.T @ r`` (``jax.vjp`` at 0), so every cycler
that only does ``Hs[i] @ x`` (Var4DOperator, ETKF4D, LETKF4D, metrics) works
unchanged and no ``(obs_dim, system_dim)`` array is ever formed.
"""

from typing import Any, Callable

import jax
import jax.numpy as jnp


@jax.tree_util.register_pytree_node_class
class LinearObsOperator:
    """Matrix-free linear observation operator.

    Args:
        fn: JAX-traceable LINEAR map ``(system_dim,) -> (obs_dim,)``
            (fixed coefficients; ``fn(0) == 0``).
        shape: ``(obs_dim, system_dim)``.
        dtype: Nominal dtype (the matrix it stands for).  ``H @ x`` returns
            ``result_type(dtype, x.dtype)``, exactly as a matrix of that
            dtype would.

    ``H @ x`` with ``x`` of shape ``(system_dim,)`` or ``(system_dim, k)``;
    ``H.T @ r`` the adjoint (``jax.vjp`` at 0).  A pytree with no
    leaves (the function and metadata are static aux data), so it passes
    through ``jit`` / ``scan`` closures like a constant.
    """

    def __init__(self, fn: Callable, shape: tuple, dtype: Any = jnp.float64):
        self.fn = fn
        self.shape = (int(shape[0]), int(shape[1]))
        self.dtype = jnp.dtype(dtype)
        self.ndim = 2

    def _apply1(self, x: jax.Array) -> jax.Array:
        return jnp.asarray(self.fn(x))

    def __matmul__(self, x: Any) -> jax.Array:
        x = jnp.asarray(x)
        dt = jnp.result_type(self.dtype, x.dtype)
        x = x.astype(dt)
        if x.ndim == 1:
            return self._apply1(x).astype(dt)
        if x.ndim == 2:
            return jax.vmap(self._apply1, in_axes=1, out_axes=1)(x).astype(dt)
        raise ValueError(f"LinearObsOperator @ x: x.ndim={x.ndim} not in (1, 2)")

    def rmatvec(self, r: Any) -> jax.Array:
        """``H^T r`` for ``r`` of shape ``(obs_dim,)`` or ``(obs_dim, k)``."""
        r = jnp.asarray(r)
        dt = jnp.result_type(self.dtype, r.dtype)
        r = r.astype(dt)
        # vjp at 0 (exact for a linear map); jax.linear_transpose would be
        # leaner but has no transpose rule for some primitives (scatter
        # without unique indices) that appear in real spectral operators.
        _, vjp = jax.vjp(self._apply1, jnp.zeros((self.shape[1],), dt))

        def one(v):
            return vjp(v.astype(dt))[0].astype(dt)
        if r.ndim == 1:
            return one(r)
        return jax.vmap(one, in_axes=1, out_axes=1)(r)

    @property
    def T(self) -> "_AdjointObsOperator":
        return _AdjointObsOperator(self)

    def astype(self, dtype: Any) -> "LinearObsOperator":
        return LinearObsOperator(self.fn, self.shape, dtype)

    def dense(self) -> jax.Array:
        """Materialise the ``(obs_dim, system_dim)`` matrix (tests / small
        diagnostics only)."""
        return self @ jnp.eye(self.shape[1], dtype=self.dtype)

    def tree_flatten(self):
        return (), (self.fn, self.shape, self.dtype)

    @classmethod
    def tree_unflatten(cls, aux, children):
        fn, shape, dtype = aux
        return cls(fn, shape, dtype)


class _AdjointObsOperator:
    def __init__(self, op: LinearObsOperator):
        self.op = op
        self.shape = (op.shape[1], op.shape[0])
        self.dtype = op.dtype

    def __matmul__(self, r: Any) -> jax.Array:
        return self.op.rmatvec(r)


def is_function_operator(H: Any) -> bool:
    return isinstance(H, LinearObsOperator)


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
        self.H = H if is_function_operator(H) else jnp.asarray(H)
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
        H = (self.H.dense() if is_function_operator(self.H) else self.H)
        Hs = jnp.broadcast_to(H[None], self.shape)
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
    if isinstance(Hs, (PerTimeH, LinearObsOperator)):
        return Hs.astype(dtype)
    return jnp.asarray(Hs, dtype)


def r_inv_diagonal(R: Any, obs_dim: int, obs_error_sd: Any) -> jax.Array:
    """Diagonal of ``R^{-1}`` as an ``(obs_dim,)`` vector, never forming an
    ``(obs_dim, obs_dim)`` matrix for diagonal R.

    ``R`` None -> ``1 / obs_error_sd**2`` (scalar or per-obs vector),
    broadcast; ``R`` 1-D -> per-obs VARIANCE vector, ``1 / R``; ``R`` 2-D ->
    an explicit dense R from the caller, ``diag(inv(R))`` (legacy).
    """
    if R is None:
        sd = jnp.asarray(obs_error_sd)
        return jnp.broadcast_to(1.0 / sd ** 2, (int(obs_dim),))
    R = jnp.asarray(R)
    if R.ndim <= 1:
        return jnp.broadcast_to(1.0 / R, (int(obs_dim),))
    import jax.scipy as jscipy
    return jnp.diag(jscipy.linalg.inv(R))
