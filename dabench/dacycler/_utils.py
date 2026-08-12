"""Utils for data assimilation cyclers"""

import jax.numpy as jnp
import jax
import numpy as np
import xarray as xr
from dabench import _xarray_jax as xj


# For typing
ArrayLike = list | np.ndarray | jax.Array
XarrayDatasetLike = xr.Dataset | xj.XjDataset

def _get_all_times(
        start_time: float,
        analysis_window: float,
        analysis_cycles: int
        ) -> jax.Array:
    """Calculate times of the centers of all analysis windows.

    Args:
        start_time: Start time of DA experiment in model time units.
        analysis_window: Length of analysis window, in model time 
            units.
        analysis_cycles: Number of analysis cycles to perform.

    Returns:
        Array of all analysis window center-times.

    
    """
    all_times = (
            jnp.repeat(start_time, analysis_cycles)
            + jnp.arange(0, analysis_cycles*analysis_window,
                            analysis_window)
                    )

    return all_times


def _get_obs_indices(
        analysis_times: ArrayLike,
        obs_times: ArrayLike,
        analysis_window: float,
        start_inclusive: bool = True,
        end_inclusive: bool = False
        ) -> list:
    """Get indices of obs times for each analysis cycle to pass to jax.lax.scan

    Args:
        analysis_times: List of times for all analysis window, centered
            in middle of time window. Output of _get_all_times().
        obs_times: List of times for all observations.
        analysis_window: Length of analysis window.
        start_inclusive: Include obs times equal to beginning of 
            analysis window. Default is True
        end_inclusive: Include obs times equal to end of 
            analysis window. Default is False.
    
    Returns:
        List with each element containing array of obs indices for the
            corresponding analysis cycle.
    """
    # Get the obs vectors for each analysis window
    all_filtered_idx = [jnp.where(
        # Greater than start of window
        (obs_times > cur_time - analysis_window/2)
        # AND Less than end of window
        * (obs_times < cur_time + analysis_window/2)
        # AND not equal to start of window
        * (1-(1-start_inclusive)*jnp.isclose(obs_times, cur_time - analysis_window/2,
                                             rtol=0))
        # AND not equal to end of window
        * (1-(1-end_inclusive)*jnp.isclose(obs_times, cur_time + analysis_window/2,
                                           rtol=0))
        # OR Equal to start of window end
        + start_inclusive*jnp.isclose(obs_times, cur_time - analysis_window/2,
                                      rtol=0)
        # OR Equal to end of window
        + end_inclusive*jnp.isclose(obs_times, cur_time + analysis_window/2,
                                    rtol=0)
        )[0] for cur_time in analysis_times]

    return all_filtered_idx


def _time_resize(
        row: ArrayLike,
        size: int,
        add_one: bool
        ) -> np.ndarray:
    new = np.array(row) + add_one
    new.resize(size)
    return new


def _pad_time_indices(
        obs_indices: ArrayLike,
        add_one: bool = True
        ) -> ArrayLike:
    """Pad observation indices for each analysis window.

    Args:
        obs_indices: List of arrays where each array contains
            obs indices for an analysis cycle. Result of _get_obs_indices.
        add_one: If True, will add one to all index values to encode
            indices to be masked out for DA (i.e. zeros represent indices to
            be masked out). Default is True.

    Returns:
        padded_indices: Array of padded obs_indices, with shape: 
            (num_analysis_cycles, max_obs_per_cycle).
    """
    # find longest row length
    row_length = max(obs_indices, key=len).__len__()
    padded_indices = np.array([_time_resize(row, row_length, add_one)
                               for row in obs_indices])

    return padded_indices


def _obs_resize(
        row: ArrayLike,
        size: float
        ) -> np.ndarray:
    new_vals_locs = np.array(np.stack(row), order='F')
    new_vals_locs.resize((new_vals_locs.shape[0], size))
    mask = np.ones_like(new_vals_locs[0]).astype(int)
    if size > len(row[0]):
        mask[-(size-len(row[0])):] = 0
    return np.vstack([new_vals_locs, mask]).T


def _pad_obs_locs(
        obs_vec: XarrayDatasetLike
        ) -> tuple[ArrayLike, ArrayLike, ArrayLike]:
    """Pad observation location indices to equal spacing

    Args:
        obs_vec: Xarray containing times, locations, and values of obs.

    Returns:
        Tuple containing padded arrays of obs
            values and locations, and binary array masks where 1 is
            a valid observation value/location and 0 is not.
    """
    # Find longest row length
    row_length = max(obs_vec.values, key=len).__len__()
    padded_arrays_masks = np.array([_obs_resize(row, row_length) for row in
                                    np.stack([obs_vec.values,
                                              obs_vec.location_indices],
                                              axis=1)], dtype=float)
    vals, locs, masks = (padded_arrays_masks[...,0],
                         padded_arrays_masks[...,1:-1].astype(int),
                         padded_arrays_masks[...,2].astype(bool))
    if locs.shape[-1] == 1:
        locs = locs[..., 0]

    return vals, locs, masks


# ── Shared SPD linear algebra for the ETKF transform ──────────────────────
# Used by ETKF / ETKF4D (one global K x K solve) and LETKF / LETKF4D (one per
# grid point, under vmap).  Kept here (a cycler-agnostic module) so both the
# base ETKF and its LETKF subclass import the SAME solver without a circular
# import through the package __init__.

def _spd_inv_sqrt_ns(A: ArrayLike, n_iter: int = 20) -> jax.Array:
    """Inverse square root ``A^{-1/2}`` of an SPD matrix via Newton-Schulz.

    Coupled (product-form Denman-Beavers) iteration -- **matmul-only**, so XLA
    lowers it to batched GEMM on the accelerator, sidestepping the eigh/SVD
    kernels that XLA runs UNBATCHED (sequentially, on the host) for matrices
    wider than 32.  For the SPD ETKF transform this matches the eigh-based
    ``A^{-1/2}`` to round-off and is what makes the T42 LETKF solve GPU-bound.

    Scale ``B = A / s`` with ``s`` an UPPER BOUND on the spectral radius (row-
    sum / Gershgorin bound) so every eigenvalue of ``B`` lies in ``(0, 1]`` --
    the convergence region.  Iterate::

        Y_0 = B, Z_0 = I
        T   = 1.5 I - 0.5 Z_k Y_k
        Y_{k+1} = Y_k T,  Z_{k+1} = T Z_k

    ``Y_k -> B^{1/2}``, ``Z_k -> B^{-1/2}`` quadratically; undo the scale:
    ``A^{-1/2} = Z_inf / sqrt(s)``.  Operates on the last two axes, so it
    ``vmap``/batches cleanly.

    Args:
        A: SPD matrix (or batch), ``(..., K, K)``.
        n_iter: Coupled iterations (fp64 64x64, kappa<1e3: ~15 -> ~1e-12).

    Returns:
        ``A^{-1/2}``, same shape/dtype as ``A``.
    """
    A = jnp.asarray(A)
    dtype = A.dtype
    K = A.shape[-1]
    I = jnp.eye(K, dtype=dtype)
    # Gershgorin upper bound on the spectral radius: max absolute row sum.
    # (>= rho(A) for any A; tight enough that B=A/s has eigenvalues in (0,1].)
    s = jnp.max(jnp.sum(jnp.abs(A), axis=-1), axis=-1)         # (...,)
    s = jnp.maximum(s, jnp.asarray(jnp.finfo(dtype).tiny, dtype))
    s = s[..., None, None]
    Y = A / s
    Z = jnp.broadcast_to(I, A.shape).astype(dtype)
    half = jnp.asarray(0.5, dtype)
    three_half = jnp.asarray(1.5, dtype)
    for _ in range(int(n_iter)):
        T = three_half * I - half * (Z @ Y)
        Y = Y @ T
        Z = T @ Z
    return Z / jnp.sqrt(s)


def _resolve_eigh_impl(eigh_impl: str | None) -> str | None:
    """Validate/normalize an ``eigh_impl`` selector (shared by all cyclers).

    Accepts ``None``/``"qr"``/``"jacobi"``/``"newton_schulz"``/``"ns"`` (the
    last folds to ``"newton_schulz"``); anything else raises so a typo fails
    loudly at construction rather than silently defaulting.
    """
    _valid = ("qr", "jacobi", "newton_schulz", "ns")
    if eigh_impl is not None and str(eigh_impl).lower() not in _valid:
        raise ValueError(
            "eigh_impl must be None, 'qr', 'jacobi', or 'newton_schulz', "
            f"got {eigh_impl!r}")
    _ei = None if eigh_impl is None else str(eigh_impl).lower()
    return "newton_schulz" if _ei == "ns" else _ei


def _solve_pa_wa(A: ArrayLike, eigh_impl: str | None, ns_iters: int
                 ) -> tuple[jax.Array, jax.Array, jax.Array]:
    """Shared SPD solve ``Pa = A^{-1}``, ``Wa = ((K-1) A^{-1})^{1/2}``.

    The single source of truth for the ETKF transform's SPD linear algebra,
    used by :class:`ETKF`, :class:`ETKF4D` (one global ``K x K`` solve) and
    :class:`LETKF`/:class:`LETKF4D` (one per grid point, under ``vmap``).
    ``A = (K-1)/rho I + Y^T R^{-1} Y`` is SPD.

    Backends (all matching to round-off for SPD ``A``):
      * eigh-based (``None``/``qr``/``jacobi``): symmetric eigendecomp,
        ``Pa=(V/w)V^T``, ``Wa=(V sqrt((K-1)/w))V^T``.  ``None`` uses
        ``jnp.linalg.eigh`` (byte-identical to prior callers); the lax path
        swaps its ``(v, w)`` return to jnp's ``(w, v)``.
      * ``newton_schulz``: eigh-FREE.  ``_spd_inv_sqrt_ns`` computes
        ``Z = A^{-1/2}`` via a matmul-only coupled iteration (batched GEMM on
        the accelerator), then ``Pa = Z Z``, ``Wa = sqrt(K-1) Z``.

    Returns ``(Pa, Wa, ns_resid)`` where ``ns_resid`` is the relative
    convergence residual ``||Z A Z - I||_F / sqrt(K)`` on the NS path (a
    finite scalar signalling whether ``ns_iters`` sufficed), and ``NaN`` on
    the eigh paths (no iterative residual to report).
    """
    A = jnp.asarray(A)
    dtype = A.dtype
    K = A.shape[-1]
    I = jnp.eye(K, dtype=dtype)
    eigh_impl = _resolve_eigh_impl(eigh_impl)
    if eigh_impl == "newton_schulz":
        Z = _spd_inv_sqrt_ns(A, n_iter=int(ns_iters))          # A^{-1/2}
        Pa = Z @ Z                                             # A^{-1}
        Wa = jnp.sqrt(jnp.asarray(K - 1, dtype)) * Z           # ((K-1)A^{-1})^½
        # Relative residual of the inverse-sqrt: ||Z A Z - I||_F / sqrt(K).
        resid = (jnp.linalg.norm(Z @ A @ Z - I)
                 / jnp.sqrt(jnp.asarray(K, dtype)))
        return Pa, Wa, resid
    if eigh_impl is None:
        w_eig, V = jnp.linalg.eigh(A)
    else:
        # EighImplementation values are lowercase ("qr"/"jacobi"); the lax API
        # returns (v, w) -- swap to jnp's (w, v).
        impl = jax.lax.linalg.EighImplementation(eigh_impl)
        V, w_eig = jax.lax.linalg.eigh(A, implementation=impl)
    inv_eig = jnp.where(w_eig > 0, 1.0 / w_eig, 0.0)
    Pa = (V * inv_eig) @ V.T                                   # pinv(A), SPD
    sqrt_eig = jnp.sqrt(jnp.clip((K - 1) * inv_eig, 0.0))
    Wa = (V * sqrt_eig) @ V.T                                  # ((K-1)A^{-1})^½
    return Pa, Wa, jnp.asarray(jnp.nan, dtype)