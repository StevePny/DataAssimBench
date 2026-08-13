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


# ── Shared observation-space DA metrics reduction ─────────────────────────
# Single source of truth for the native obs-space diagnostics emitted (opt-in)
# by every cycler's cycle(..., return_metrics=True) path.  All cyclers call
# THIS implementation so the reduction (masking, RMS, bias, spread) is
# identical across ETKF / ETKF4D / LETKF / LETKF4D / Var4D / Var4DOperator.

def _b_derived_obs_spread(H, C, active_mask, dtype=None):
    """RMS per-obs obs-space std implied by a state-space covariance ``C``.

    Deterministic (variational) analogue of the ensemble ``obs_space_spread``:
    projects ``C`` into obs space (``diag(H C H^T)``, the per-obs prior/posterior
    variance) and returns ``sqrt(mean_over_active(diag))``.  ``H`` is the
    observation operator (obs x state), ``C`` a state-space covariance (e.g. the
    static background ``B`` or the analysis posterior ``A``).  NaN when no active
    obs.  Used by Var3D/Var4D to emit the B-derived spread metrics in place of
    the ensemble estimate (which they do not have).
    """
    dtype = H.dtype if dtype is None else dtype
    H = jnp.asarray(H, dtype)
    C = jnp.asarray(C, dtype)
    active = jnp.asarray(active_mask, bool)
    m = active.astype(dtype)
    n = jnp.sum(m)
    per_obs_var = jnp.sum((H @ C) * H, axis=1)            # diag(H C H^T)
    safe = jnp.where(n > 0, n, jnp.asarray(1, dtype))
    return jnp.where(n > 0, jnp.sqrt(jnp.sum(per_obs_var * m) / safe),
                     jnp.asarray(jnp.nan, dtype))


def _obs_space_metrics(y, Hxb_mean, Hxa_mean, active_mask, sigma2_diag,
                       ens_obs=None, return_per_obs=False, dtype=None,
                       Hxa_end_mean=None, end_active_mask=None,
                       ens_obs_end=None, spread_background_override=None,
                       spread_analysis_end_override=None):
    """Aggregate (+ optional per-obs) observation-space DA metrics.

    All reductions are restricted to ACTIVE observations (``active_mask``);
    a cycle with zero active obs yields NaN scalars and ``n_active_obs=0``.
    Innovations are vs the actual observations ``y`` (O-F = y - Hxb_mean,
    O-A = y - Hxa_mean).  ``obs_space_spread_background`` (the PRIOR/background
    spread at the analysis time) is the RMS per-obs ensemble std when
    ``ens_obs`` is supplied; deterministic (variational) cyclers instead pass a
    precomputed B-derived scalar via ``spread_background_override`` (see
    :func:`_b_derived_obs_spread`).  It is NaN only when neither is supplied.
    ``dtype`` follows the analysis dtype (fp64 x64).

    End-of-window O-A (the quality of the ICs handed to the NEXT cycle):
    ``Hxa_end_mean`` is the analysis mean PROPAGATED TO THE WINDOW END, and
    ``end_active_mask`` selects the observations valid at the window end (obs
    whose true time == the window end).  When supplied, ``o_minus_a_rms_end``
    /``bias_a_end`` score ``y - Hxa_end_mean`` over the END-obs only; when
    omitted they are NaN.  These two scalars are ALWAYS present in the returned
    dict (NaN when not supplied) so the emitted metric schema is uniform across
    every cycler (a ``jax.lax.scan`` pytree-consistency requirement).

    End-of-window ANALYSIS spread (the spread of the ensemble handed to the
    NEXT cycle -- distinct from ``obs_space_spread_background``, which is the
    PRIOR spread at the analysis time ``tau``): ``ens_obs_end`` is the
    analysis-ensemble obs-space PERTURBATIONS (obs x ens) at the window end.
    When supplied, ``obs_space_spread_analysis_end`` is the RMS per-obs ensemble
    std over the END-obs (``end_active_mask``).  Deterministic (variational)
    cyclers instead pass a precomputed posterior scalar via
    ``spread_analysis_end_override`` (Var4D: the TLM-propagated posterior at the
    window end; Var3D-FGAT: the static posterior at ``tau``).  NaN only when
    neither is supplied.  Always present in the returned dict for schema
    uniformity.
    """
    dtype = y.dtype if dtype is None else dtype
    y = jnp.asarray(y, dtype)
    active = jnp.asarray(active_mask, bool)
    m = active.astype(dtype)
    n_active = jnp.sum(m)
    of = y - jnp.asarray(Hxb_mean, dtype)
    oa = y - jnp.asarray(Hxa_mean, dtype)
    of_m, oa_m = of * m, oa * m
    safe = jnp.where(n_active > 0, n_active, jnp.asarray(1, dtype))
    of_rms = jnp.sqrt(jnp.sum(of_m ** 2) / safe)
    oa_rms = jnp.sqrt(jnp.sum(oa_m ** 2) / safe)
    bias_f = jnp.sum(of_m) / safe
    bias_a = jnp.sum(oa_m) / safe
    if spread_background_override is not None:
        # Deterministic (variational) B-derived background spread; already a
        # reduced scalar (no per-obs / active masking to reapply here).
        spread = jnp.asarray(spread_background_override, dtype)
    elif ens_obs is None:
        spread = jnp.asarray(jnp.nan, dtype)
    else:
        per_obs_var = jnp.mean(jnp.asarray(ens_obs, dtype) ** 2, axis=1)
        spread = jnp.sqrt(jnp.sum(per_obs_var * m) / safe)
    s2a = jnp.where(active, jnp.asarray(sigma2_diag, dtype),
                    jnp.asarray(-jnp.inf, dtype))
    sigma_obs_max = jnp.sqrt(jnp.max(s2a))
    nan = jnp.asarray(jnp.nan, dtype)
    valid = n_active > 0

    # End-of-window O-A (next-cycle IC quality), scored over END-obs only.
    if Hxa_end_mean is None:
        oa_end_rms = nan
        bias_a_end = nan
        n_end = jnp.asarray(0, dtype)
    else:
        end_active = (active if end_active_mask is None
                      else jnp.asarray(end_active_mask, bool))
        me = end_active.astype(dtype)
        n_end = jnp.sum(me)
        safe_e = jnp.where(n_end > 0, n_end, jnp.asarray(1, dtype))
        oa_e = (y - jnp.asarray(Hxa_end_mean, dtype)) * me
        oa_end_rms = jnp.where(n_end > 0,
                               jnp.sqrt(jnp.sum(oa_e ** 2) / safe_e), nan)
        bias_a_end = jnp.where(n_end > 0, jnp.sum(oa_e) / safe_e, nan)

    # End-of-window ANALYSIS spread (next-cycle IC spread), over END-obs only;
    # emitted as ``obs_space_spread_analysis_end`` (cf. the PRIOR/background
    # spread at tau emitted as ``obs_space_spread_background``).
    if spread_analysis_end_override is not None:
        # Deterministic (variational) posterior spread scalar (Var4D:
        # TLM-propagated to the window end; Var3D-FGAT: static posterior at tau).
        spread_end = jnp.asarray(spread_analysis_end_override, dtype)
    elif ens_obs_end is None:
        spread_end = nan
    else:
        end_active_s = (active if end_active_mask is None
                        else jnp.asarray(end_active_mask, bool))
        me_s = end_active_s.astype(dtype)
        n_end_s = jnp.sum(me_s)
        safe_es = jnp.where(n_end_s > 0, n_end_s, jnp.asarray(1, dtype))
        per_obs_var_e = jnp.mean(jnp.asarray(ens_obs_end, dtype) ** 2, axis=1)
        spread_end = jnp.where(
                n_end_s > 0,
                jnp.sqrt(jnp.sum(per_obs_var_e * me_s) / safe_es), nan)

    out = {
        "o_minus_f_rms": jnp.where(valid, of_rms, nan),
        "o_minus_a_rms": jnp.where(valid, oa_rms, nan),
        "bias_f": jnp.where(valid, bias_f, nan),
        "bias_a": jnp.where(valid, bias_a, nan),
        "obs_space_spread_background": (
                spread if (ens_obs is None
                           or spread_background_override is not None)
                else jnp.where(valid, spread, nan)),
        "sigma_obs_max": jnp.where(valid, sigma_obs_max, nan),
        "n_active_obs": n_active.astype(dtype),
        "o_minus_a_rms_end": oa_end_rms,
        "bias_a_end": bias_a_end,
        "n_active_obs_end": n_end,
        "obs_space_spread_analysis_end": spread_end,
    }
    if return_per_obs:
        out["o_minus_f"] = jnp.where(active, of, nan)
        out["o_minus_a"] = jnp.where(active, oa, nan)
        out["obs_active"] = m
    return out


# ── Reshapeable, after-run-accessible metrics container ───────────────────
class CyclerMetrics:
    """Reshapeable container for a cycler's per-cycle obs-space DA metrics.

    Wraps the assembled per-cycle metrics (one row per analysis cycle) and is
    stored on the cycler instance after ``cycle(..., return_metrics=True)`` (as
    ``cycler.metrics``), so the diagnostics are accessible AFTER the run without
    having to thread the return tuple through every caller.  It is also still
    returned from ``cycle`` for backward compatibility.

    Backed by an internal :class:`xarray.Dataset` so it duck-types the previous
    ``metrics_ds[name].data`` / ``name in metrics_ds`` usage exactly (existing
    consumers need no change), while adding reshape / conversion / memory
    helpers:

    * ``to_dataset()``      -> the underlying :class:`xarray.Dataset`.
    * ``to_numpy()``        -> ``{name: np.ndarray}`` flat dict.
    * ``as_dict()``         -> alias of ``to_numpy()``.
    * ``keys()``            -> metric names.
    * ``reshape(**shape)``  -> a NEW CyclerMetrics with the leading ``cycle``
      axis reshaped (e.g. ``reshape(outer=n_outer, inner=n_inner)`` to split a
      flat cycle axis into a 2-D layout).
    * ``sel_cycles(sl)``    -> a NEW CyclerMetrics restricted to a cycle slice.
    * ``nbytes``            -> total array bytes held (memory stat).
    * ``memory_report()``   -> human-readable one-line memory/shape summary.

    Memory: in the default metrics mode only per-cycle SCALARS are held (tiny),
    so this is negligible even at high resolution; per-obs debug arrays
    (``metrics_mode='debug'``) are the only growth term and are surfaced via
    :attr:`nbytes` / :meth:`memory_report` so callers can watch usage at T42+.
    """

    def __init__(self, dataset: xr.Dataset):
        self._ds = dataset

    # -- duck-typing the previous xr.Dataset usage --------------------------
    def __getitem__(self, key):
        return self._ds[key]

    def __contains__(self, key):
        return key in self._ds

    def __iter__(self):
        return iter(self._ds)

    def keys(self):
        return list(self._ds.data_vars)

    @property
    def sizes(self):
        return self._ds.sizes

    # -- conversions --------------------------------------------------------
    def to_dataset(self) -> xr.Dataset:
        return self._ds

    def to_numpy(self) -> dict:
        return {k: np.asarray(self._ds[k].data) for k in self._ds.data_vars}

    def as_dict(self) -> dict:
        return self.to_numpy()

    # -- reshaping ----------------------------------------------------------
    def reshape(self, **shape) -> "CyclerMetrics":
        """Split the leading ``cycle`` axis into named dims (product must match).

        Example: ``m.reshape(experiment=3, cycle=10)`` turns a length-30 cycle
        axis into a ``(experiment, cycle)`` layout on every metric variable.
        """
        n = int(self._ds.sizes.get("cycle", 0))
        prod = int(np.prod(list(shape.values()))) if shape else 0
        if prod != n:
            raise ValueError(
                f"reshape product {prod} does not match cycle length {n}")
        new_dims = tuple(shape.keys())
        data_vars = {}
        for k in self._ds.data_vars:
            da = self._ds[k]
            trailing = tuple(d for d in da.dims if d != "cycle")
            arr = np.asarray(da.data).reshape(
                    tuple(shape.values()) + tuple(
                        da.sizes[d] for d in trailing))
            data_vars[k] = (new_dims + trailing, arr)
        return CyclerMetrics(xr.Dataset(data_vars))

    def sel_cycles(self, cycle_slice) -> "CyclerMetrics":
        return CyclerMetrics(self._ds.isel(cycle=cycle_slice))

    # -- memory -------------------------------------------------------------
    @property
    def nbytes(self) -> int:
        return int(sum(np.asarray(self._ds[k].data).nbytes
                       for k in self._ds.data_vars))

    def memory_report(self) -> str:
        mb = self.nbytes / (1024.0 ** 2)
        n_cycles = int(self._ds.sizes.get("cycle", 0))
        n_vars = len(self._ds.data_vars)
        has_obs = "obs" in self._ds.sizes
        obs = f", obs={int(self._ds.sizes.get('obs', 0))}" if has_obs else ""
        return (f"CyclerMetrics: {n_vars} vars, cycles={n_cycles}{obs}, "
                f"{mb:.3f} MiB")

    def __repr__(self) -> str:
        return f"<{self.memory_report()}>"