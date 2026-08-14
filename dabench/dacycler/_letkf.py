"""Class for Local Ensemble Transform Kalman Filter (LETKF) DA Cycler."""

import numpy as np
import jax
import jax.numpy as jnp
import xarray as xr
from dabench import _xarray_jax as xj
from typing import Callable

from dabench.dacycler import ETKF
from dabench.dacycler._utils import (  # noqa: F401  (re-exported for callers)
    _resolve_eigh_impl, _solve_pa_wa, _spd_inv_sqrt_ns)


# For typing
ArrayLike = np.ndarray | jax.Array
XarrayDatasetLike = xr.Dataset | xj.XjDataset

_EARTH_RADIUS_KM = 6371.0
_DEG2KM = 111.195


def _gaspari_cohn(dist: ArrayLike, c: float) -> jax.Array:
    """Gaspari-Cohn (1999) 5th-order piecewise-rational taper.

    Compactly-supported correlation function: 1 at ``dist=0``, C1-smooth,
    and exactly 0 beyond ``2 * c`` (``c`` = the taper half-width / "sigma
    radius").  Used for R-localization (tapers each observation's inverse
    error variance by distance from the analysis grid point).

    Args:
        dist: Distances (same units as ``c``), any shape.
        c: Taper half-width; the support cutoff is ``2 * c``.

    Returns:
        Taper weights in ``[0, 1]``, same shape as ``dist``.
    """
    dtype = jnp.asarray(dist).dtype
    r = jnp.abs(jnp.asarray(dist)) / jnp.asarray(c, dtype=dtype)
    r2, r3, r4, r5 = r ** 2, r ** 3, r ** 4, r ** 5
    # 0 <= r <= 1
    near = (-0.25 * r5 + 0.5 * r4 + 0.625 * r3 - (5.0 / 3.0) * r2 + 1.0)
    # 1 < r <= 2
    far = (r5 / 12.0 - 0.5 * r4 + 0.625 * r3 + (5.0 / 3.0) * r2
           - 5.0 * r + 4.0 - (2.0 / 3.0) / jnp.where(r > 0, r, 1.0))
    w = jnp.where(r <= 1.0, near, jnp.where(r <= 2.0, far, 0.0))
    return jnp.clip(w, 0.0, 1.0).astype(dtype)


def _great_circle_km(lat1: ArrayLike, lon1: ArrayLike,
                     lat2: ArrayLike, lon2: ArrayLike) -> jax.Array:
    """Great-circle distance (km) between (lat1,lon1) and (lat2,lon2) in deg.

    Broadcasts over its inputs (haversine formula, Earth radius 6371 km).
    """
    la1, lo1 = jnp.deg2rad(lat1), jnp.deg2rad(lon1)
    la2, lo2 = jnp.deg2rad(lat2), jnp.deg2rad(lon2)
    dlat = la2 - la1
    dlon = lo2 - lo1
    a = (jnp.sin(dlat / 2.0) ** 2
         + jnp.cos(la1) * jnp.cos(la2) * jnp.sin(dlon / 2.0) ** 2)
    return 2.0 * _EARTH_RADIUS_KM * jnp.arcsin(jnp.sqrt(jnp.clip(a, 0.0, 1.0)))


def build_patch_geometry(grid_latlon: ArrayLike,
                         obs_latlon: ArrayLike,
                         localize_radius: float,
                         localize_units: str = "km",
                         patch_size: int | None = None,
                         ) -> tuple[np.ndarray, np.ndarray, dict]:
    """Static local-patch geometry for Regime-A LETKF localization.

    Replaces the dense ``(grid_dim, n_obs)`` Gaspari-Cohn taper (whose
    per-lane ``(K, n_obs)`` intermediate drives the T42-4D OOM) with a sparse
    ``(grid_dim, P)`` gather over each grid point's ``<= P`` nearest pool
    observations.  The obs network is a FIXED POOL, so this geometry is static
    and is built ONCE, host-side, at setup; only the per-cycle activity mask
    changes on-device (see :meth:`LETKF._build_patch_w`).

    Distances and Gaspari-Cohn weights use the SAME great-circle metric and
    ``_gaspari_cohn`` formula as the dense taper, so gathering is a strict
    refinement: at ``P = max neighbour count`` the patch analysis matches the
    dense-taper analysis to round-off (regression-tested).

    Args:
        grid_latlon: ``(grid_dim, 2)`` (lat_deg, lon_deg) per grid point.
        obs_latlon: ``(n_pool, 2)`` (lat_deg, lon_deg) per pool observation
            (the single, stationary obs slice; NOT window-stacked).
        localize_radius: Gaspari-Cohn half-width ``c`` (support cutoff ``2c``).
        localize_units: ``"km"`` (default) or ``"deg"`` (converted via 111.195).
        patch_size: Fixed ``P``.  ``None`` -> the exact ``max`` neighbour count
            within ``2c`` (no truncation).  A smaller ``P`` keeps the ``P``
            NEAREST obs per over-full grid point (dropping the farthest, whose
            GC weight is smallest) -- a controlled, logged approximation.

    Returns:
        ``(patch_idx, patch_gc, diag)``:
          * ``patch_idx``: ``(grid_dim, P)`` int32, pool-obs indices of the
            ``<= P`` nearest obs (padding slots filled with index 0).
          * ``patch_gc``: ``(grid_dim, P)`` float64, distance-only Gaspari-Cohn
            weights (0 for padding slots AND for obs beyond ``2c``).
          * ``diag``: histogram diagnostics -- ``P``, ``max/median/p99`` patch
            size, ``%`` grid points truncated, and the fraction of GC weight
            mass discarded by truncation.
    """
    grid_ll = np.asarray(grid_latlon, dtype=np.float64)
    obs_ll = np.asarray(obs_latlon, dtype=np.float64)
    if grid_ll.ndim != 2 or grid_ll.shape[1] != 2:
        raise ValueError(f"grid_latlon must be (grid_dim, 2); got {grid_ll.shape}")
    if obs_ll.ndim != 2 or obs_ll.shape[1] != 2:
        raise ValueError(f"obs_latlon must be (n_pool, 2); got {obs_ll.shape}")
    if localize_units not in ("km", "deg"):
        raise ValueError(f"localize_units must be 'km' or 'deg'; got {localize_units!r}")
    c_km = float(localize_radius) * (1.0 if localize_units == "km" else _DEG2KM)
    cutoff = 2.0 * c_km
    G = grid_ll.shape[0]

    # Great-circle distances (km) grid -> pool, matching the dense taper metric.
    dist = np.asarray(_great_circle_km(
        grid_ll[:, 0][:, None], grid_ll[:, 1][:, None],
        obs_ll[:, 0][None, :], obs_ll[:, 1][None, :]))          # (grid_dim, n_pool)

    within = dist <= cutoff
    n_within = within.sum(axis=1).astype(np.int64)              # (grid_dim,)
    P_exact = int(n_within.max()) if n_within.size else 0
    P_exact = max(P_exact, 1)                                   # avoid empty axis
    P = P_exact if patch_size is None else int(patch_size)
    if P < 1:
        raise ValueError(f"patch_size must be >= 1; got {P}")

    # For each grid point keep the P nearest obs (argpartition then sort the
    # P-slice by distance for deterministic ordering); mask any beyond 2c with
    # weight 0, and any padding slots (fewer than P neighbours) with weight 0.
    if P >= dist.shape[1]:
        order = np.argsort(dist, axis=1)                       # all obs, sorted
        idx = order[:, :P]
        if idx.shape[1] < P:                                   # pad short axis
            pad = P - idx.shape[1]
            idx = np.pad(idx, ((0, 0), (0, pad)))
    else:
        part = np.argpartition(dist, P - 1, axis=1)[:, :P]     # P nearest (unord)
        rows = np.arange(G)[:, None]
        part_sorted = np.argsort(dist[rows, part], axis=1)
        idx = part[rows, part_sorted]                          # (grid_dim, P)

    rows = np.arange(G)[:, None]
    patch_dist = dist[rows, idx]                               # (grid_dim, P)
    patch_gc = np.asarray(_gaspari_cohn(patch_dist, c_km), dtype=np.float64)
    patch_gc = np.where(patch_dist <= cutoff, patch_gc, 0.0)   # kill >2c slots
    patch_idx = idx.astype(np.int32)

    # Truncation diagnostics: only meaningful when P < max neighbour count.
    trunc = n_within > P
    n_trunc = int(trunc.sum())
    if patch_size is not None and n_trunc > 0:
        # Weight mass kept (in-patch, <=2c) vs total mass over ALL <=2c obs.
        full_gc = np.asarray(_gaspari_cohn(dist, c_km), dtype=np.float64)
        full_gc = np.where(within, full_gc, 0.0)
        mass_total = float(full_gc.sum())
        mass_kept = float(patch_gc.sum())
        mass_discarded = (0.0 if mass_total <= 0.0
                          else 1.0 - mass_kept / mass_total)
    else:
        mass_discarded = 0.0
    diag = {
        "P": int(P),
        "P_exact": int(P_exact),
        "patch_max": int(n_within.max()) if n_within.size else 0,
        "patch_median": float(np.median(n_within)) if n_within.size else 0.0,
        "patch_p99": float(np.percentile(n_within, 99)) if n_within.size else 0.0,
        "pct_truncated": (100.0 * n_trunc / G) if G else 0.0,
        "weight_mass_discarded": float(mass_discarded),
    }
    return patch_idx, patch_gc, diag


class LETKF(ETKF):
    """Local Ensemble Transform Kalman Filter DA Cycler (Hunt et al. 2007).

    Domain-localized ETKF: the analysis is computed per *grid point* by
    tapering each observation's inverse error variance with a Gaspari-Cohn
    weight of its distance to that grid point (R-localization), then solving
    the same K x K ETKF transform on the tapered local observations.

    Localization is a **grid-space** operation.  For spectral models the
    caller injects a spectral<->grid transform pair (``to_grid`` /
    ``from_grid``); for grid-native models (e.g. Lorenz96) the identity
    transforms are used and the state IS the grid.  The great-circle taper
    is precomputed once (obs network is a fixed pool) as a dense
    ``(grid_dim, n_obs)`` matrix and reused every cycle.

    Args:
        (all :class:`ETKF` / :class:`ETKF4D` args, plus:)
        to_grid: ``spec (system_dim,) -> grid (grid_dim,)``; default identity.
        from_grid: ``grid (grid_dim,) -> spec (system_dim,)``; default identity.
        grid_latlon: ``(grid_dim, 2)`` array of (lat_deg, lon_deg) per grid
            point.  If None, a 1-D index coordinate is used (Lorenz96-style
            ring) and ``localize_units`` is treated as index units.
        obs_latlon: ``(n_obs, 2)`` (lat_deg, lon_deg) per observation.  If
            None it is gathered from ``grid_latlon`` at the obs indices.
        localize_radius: Gaspari-Cohn half-width ``c`` (support cutoff 2c).
            Default 800.
        localize_units: ``"km"`` (default) or ``"deg"`` for the radius +
            distances; ``deg`` is converted to km via 111.195.
        localize_taper: Taper name; only ``"gaspari_cohn"`` supported.
    """
    _in_4d: bool = False
    _uses_ensemble: bool = True

    def __init__(self, *args,
                 to_grid: Callable | None = None,
                 from_grid: Callable | None = None,
                 grid_latlon: ArrayLike | None = None,
                 obs_latlon: ArrayLike | None = None,
                 localize_radius: float = 800.0,
                 localize_units: str = "km",
                 localize_taper: str = "gaspari_cohn",
                 grid_chunk: int | None = 512,
                 eigh_impl: str | None = None,
                 ns_iters: int = 40,
                 ns_resid_warn: float = 1e-6,
                 increment_taper: ArrayLike | None = None,
                 patch_idx: ArrayLike | None = None,
                 patch_gc: ArrayLike | None = None,
                 **kwargs):
        self.to_grid = (lambda x: x) if to_grid is None else to_grid
        self.from_grid = (lambda x: x) if from_grid is None else from_grid
        self._grid_latlon = (None if grid_latlon is None
                             else jnp.asarray(grid_latlon))
        self._obs_latlon = (None if obs_latlon is None
                            else jnp.asarray(obs_latlon))
        # Regime-A local-patch geometry (§14.3): static ``(grid_dim, P)``
        # pool-obs indices + distance-only Gaspari-Cohn weights from
        # :func:`build_patch_geometry`.  When BOTH are supplied the solve reads
        # only each grid point's ``<= P`` nearest obs (``(grid_dim, K, P)`` peak)
        # instead of the dense ``(grid_dim, n_obs)`` taper; ``None`` (default)
        # keeps the byte-identical dense-taper path.  Both-or-neither.
        if (patch_idx is None) != (patch_gc is None):
            raise ValueError(
                "patch_idx and patch_gc must be supplied together (or both "
                "None for the dense-taper path).")
        self._patch_idx = (None if patch_idx is None
                           else jnp.asarray(patch_idx, dtype=jnp.int32))
        self._patch_gc = (None if patch_gc is None
                          else jnp.asarray(patch_gc))
        self._patch_pool_size = None
        if self._patch_idx is not None:
            if self._patch_idx.shape != self._patch_gc.shape:
                raise ValueError(
                    f"patch_idx {self._patch_idx.shape} and patch_gc "
                    f"{self._patch_gc.shape} must have the same (grid_dim, P) "
                    "shape.")
            # Pool size (single obs slice) the patch indices reference; needed to
            # map slice-local patch_idx onto the WINDOW-STACKED obs axis (§14.3).
            # Prefer the injected obs_latlon rows; else infer from max index + 1.
            self._patch_pool_size = (
                int(self._obs_latlon.shape[0]) if self._obs_latlon is not None
                else int(self._patch_idx.max()) + 1)
        self._patch_w_cache = None
        self.localize_radius = float(localize_radius)
        if localize_units not in ("km", "deg"):
            raise ValueError(
                f"localize_units must be 'km' or 'deg', got {localize_units!r}")
        self.localize_units = localize_units
        if localize_taper != "gaspari_cohn":
            raise ValueError(
                "only 'gaspari_cohn' taper is supported, got "
                f"{localize_taper!r}")
        self.localize_taper = localize_taper
        # Grid-chunk size for the per-gridpoint local solve.  The fused lane
        # materializes a ``(chunk, K, n_obs)`` intermediate; ``vmap``-ing the
        # WHOLE grid at once is ``(grid_dim, K, n_obs)`` which OOMs at T42-4D
        # (window-stacked ``n_obs`` ~ 45k -> ~150 GB).  Chunking with
        # ``jax.lax.map`` caps the peak at one block while matching the
        # whole-grid ``vmap`` to round-off (same ``_lane``, same order).
        # ``None`` -> single block (legacy whole-grid ``vmap``; small grids/L96).
        self.grid_chunk = None if grid_chunk is None else int(grid_chunk)
        # Per-gridpoint K x K SPD-solver backend for the local ETKF transform.
        # ``None`` -> XLA default eigh (QR on CPU/GPU); byte-identical to every
        # existing caller.  Alternatives (all matching the eigh path to
        # round-off for the SPD ``A``; opt-in so L96 / CPU keep ``None``):
        #   "qr"     -- lax eigh, explicit QR (== None, via the lax API).
        #   "jacobi" -- lax eigh, batched Jacobi (GPU/TPU only).  NOTE: cuSOLVER
        #               syevjBatched is capped at 32x32, so at K=64 XLA still
        #               falls back to the UNBATCHED host path on stacks without
        #               CUDA>=12.6.2 -- verify it actually reaches the GPU.
        #   "newton_schulz"/"ns" -- eigh-FREE coupled Newton-Schulz inverse
        #               square root (matmul-only -> batched GEMM on the
        #               accelerator).  The robust GPU path: XLA's default GPU
        #               eigh runs UNBATCHED for K>32 (iterating the
        #               grid x cycles batch sequentially on the host --
        #               pathologically slow at K=64), and Jacobi is 32-capped;
        #               NS sidesteps eigh entirely.  See ``_solve_pa_wa``.
        # ``eigh_impl`` / ``ns_iters`` / ``ns_resid_warn`` are owned by the base
        # ETKF (single source of truth for the shared solver); forward them
        # through ``super().__init__`` rather than setting them here.
        kwargs["eigh_impl"] = eigh_impl
        kwargs["ns_iters"] = ns_iters
        kwargs["ns_resid_warn"] = ns_resid_warn
        # Optional spectral taper applied to the ANALYSIS INCREMENT (analysis
        # minus background) in spectral/state space, ``(system_dim,)`` with
        # values in ``[floor, 1]``.  The background is preserved; only the
        # newly-added increment is attenuated per mode, so the increment injects
        # no small-scale energy above the taper's passband -- suppressing the
        # spectral-detonation-at-the-truncation-edge failure mode.  ``None``
        # disables it (byte-identical to prior callers).
        self.increment_taper = (None if increment_taper is None
                                else jnp.asarray(increment_taper))
        self._taper_cache = None
        # Per-cycle SHT-truncation / energy diagnostics (updated each analysis).
        self.trunc_power = None
        self.energy_pre = None
        self.energy_post = None
        super().__init__(*args, **kwargs)

    # ── localization taper ────────────────────────────────────────────────
    def _radius_km(self) -> float:
        """Taper half-width ``c`` in km (or index units for the 1-D case)."""
        if self._grid_latlon is None:
            return self.localize_radius        # 1-D index units (Lorenz96)
        return (self.localize_radius if self.localize_units == "km"
                else self.localize_radius * _DEG2KM)

    def _distances(self, obs_loc_indices: ArrayLike) -> jax.Array:
        """Dense ``(grid_dim, n_obs)`` distance matrix (km or index units).

        Uses great-circle distances from ``grid_latlon`` when provided; else
        falls back to a periodic 1-D index ring of length ``system_dim``
        (Lorenz96-style), so localization works with no geometry injected.

        ``n_obs`` is taken from ``obs_loc_indices`` (the actual obs axis).  In
        the 4D case the obs axis is WINDOW-STACKED (``n_times * obs_dim``) while
        the injected ``obs_latlon`` describes a single, stationary obs slice
        (``obs_dim``); the slice geometry is then tiled ``n_times`` times to
        match, since the same locations recur every window step (validity is
        carried separately by the zeroed ``rinv_diag`` entries).
        """
        obs_idx = jnp.asarray(obs_loc_indices).reshape(-1).astype(jnp.int32)
        n_obs = int(obs_idx.shape[0])
        if self._grid_latlon is not None:
            grid_ll = self._grid_latlon                       # (grid_dim, 2)
            if self._obs_latlon is not None:
                obs_ll = self._obs_latlon                      # (obs_dim, 2)
                n_slice = int(obs_ll.shape[0])
                if n_obs != n_slice:
                    if n_slice == 0 or n_obs % n_slice != 0:
                        raise ValueError(
                            f"obs axis ({n_obs}) is not an integer multiple of "
                            f"the injected obs_latlon slice ({n_slice}); cannot "
                            "align the localization taper to the window-stacked "
                            "observations.")
                    obs_ll = jnp.tile(obs_ll, (n_obs // n_slice, 1))
            else:
                obs_ll = grid_ll[obs_idx]
            return _great_circle_km(
                grid_ll[:, 0][:, None], grid_ll[:, 1][:, None],
                obs_ll[:, 0][None, :], obs_ll[:, 1][None, :])
        # 1-D periodic index ring.
        n = int(self.system_dim)
        grid_pos = jnp.arange(n, dtype=jnp.float32)[:, None]
        obs_pos = obs_idx.astype(jnp.float32)[None, :]
        d = jnp.abs(grid_pos - obs_pos)
        return jnp.minimum(d, n - d)

    def _build_taper(self, obs_loc_indices: ArrayLike,
                     dtype) -> jax.Array:
        """Dense ``(grid_dim, n_obs)`` Gaspari-Cohn taper matrix.

        The obs network is a fixed pool, so the gridpoint->obs geometry is
        static and the taper is cached across cycles.  The cache is only
        populated with a CONCRETE array: when this is first reached under a
        JAX trace (``jax.lax.scan`` in :meth:`DACycler.cycle`) the taper is
        recomputed each iteration rather than caching a tracer -- caching a
        tracer would let it escape the trace and raise
        :class:`jax.errors.UnexpectedTracerError` on the next cycle, making
        the cycler single-use.  Exactly 0 beyond ``2c``.
        """
        if self._taper_cache is not None:
            return self._taper_cache.astype(dtype)
        dist = self._distances(obs_loc_indices)
        taper = _gaspari_cohn(dist, self._radius_km())
        if not isinstance(jnp.asarray(obs_loc_indices), jax.core.Tracer):
            self._taper_cache = taper
        return taper.astype(dtype)

    @property
    def _use_patch(self) -> bool:
        """Whether the sparse local-patch gather (§14.3) is active."""
        return self._patch_idx is not None

    def _build_patch_w(self, n_obs: int, dtype
                       ) -> tuple[jax.Array, jax.Array]:
        """Window-stacked patch gather indices + weights for the local solve.

        Expands the static single-slice geometry ``(grid_dim, P)`` onto the
        (possibly window-stacked) obs axis of length ``n_obs``.  In 4D the obs
        axis repeats the fixed pool ``n_times`` times (``n_obs =
        n_times * pool``), so a grid point's patch in window block ``t`` gathers
        pool slots ``patch_idx + t * pool``; the DISTANCE-ONLY Gaspari-Cohn
        weight is identical in every block (same stationary geometry, exactly as
        the dense taper tiles).  Per-cycle observation ACTIVITY is NOT applied
        here: it is carried by the zeroed ``rinv_diag`` entries the lane gathers
        (``rinv[gathered_idx]``), matching the dense path's ``rinv * taper``.

        Returns ``(patch_idx_ws, patch_w_ws)``, both ``(grid_dim, n_times * P)``:
        int32 obs-axis indices and the dtype-cast GC weights.  Cached like the
        dense taper (concrete only; recomputed under a JAX trace so no tracer
        leaks onto ``self``).
        """
        if self._patch_w_cache is not None:
            idx_c, w_c = self._patch_w_cache
            if int(idx_c.shape[1]) == n_obs // self._patch_pool_size \
                    * self._patch_idx.shape[1]:
                return idx_c, w_c.astype(dtype)
            # Obs-axis length changed (different window stacking) -> rebuild.
            self._patch_w_cache = None
        pool = self._patch_pool_size
        if pool is None or pool <= 0:
            raise ValueError(
                "patch localization requires a known pool size; supply "
                "obs_latlon or non-empty patch_idx.")
        if n_obs % pool != 0:
            raise ValueError(
                f"obs axis ({n_obs}) is not an integer multiple of the patch "
                f"pool size ({pool}); cannot align the local patches to the "
                "window-stacked observations.")
        n_times = n_obs // pool
        # (grid_dim, P) -> (grid_dim, n_times, P) with per-block pool offset,
        # then flatten the (n_times, P) axes to (grid_dim, n_times*P).
        base = self._patch_idx[:, None, :]                     # (G, 1, P)
        offs = (jnp.arange(n_times, dtype=jnp.int32) * pool
                )[None, :, None]                               # (1, nt, 1)
        idx_ws = (base + offs).reshape(self._patch_idx.shape[0], n_times
                                       * self._patch_idx.shape[1])
        w_ws = jnp.broadcast_to(
            self._patch_gc[:, None, :],
            (self._patch_gc.shape[0], n_times, self._patch_gc.shape[1])
            ).reshape(idx_ws.shape)
        self._patch_w_cache = (idx_ws, w_ws)
        return idx_ws, w_ws.astype(dtype)

    # ── local (per-gridpoint) ETKF transform capture ──────────────────────
    def capture_A_matrices(self,
                           Xb: ArrayLike,
                           Yb: ArrayLike,
                           Y: ArrayLike,
                           rinv_diag: ArrayLike,
                           obs_loc_indices: ArrayLike,
                           rho: float) -> jax.Array:
        """Materialize the per-gridpoint SPD transforms ``A`` (no solve).

        Reproduces EXACTLY the ``A = (K-1)/rho I + Y^T R^{-1} Y`` that
        :meth:`_local_columns` builds per grid point (same ISHT lift, taper,
        window-stacked innovations), but returns the raw ``(grid_dim, K, K)``
        stack instead of solving/recombining.  Intended for OFFLINE solver
        diagnostics (replay the real transforms through different SPD backends /
        precisions / ridges).  Call EAGERLY (outside ``lax.scan``) so the result
        is concrete; the inputs are the same the cycle body passes to
        :meth:`_localized_analysis`.

        Args:
            Xb: Background ensemble in spectral/state space, ``(system_dim, K)``.
            Yb: Obs-space ensemble, ``(n_obs, K)``.
            Y: Flattened observation vector, ``(n_obs,)``.
            rinv_diag: Masked diagonal ``R^{-1}``, ``(n_obs,)``.
            obs_loc_indices: Flattened observed grid indices (for the taper).
            rho: Multiplicative inflation factor.

        Returns:
            The per-gridpoint SPD transform stack ``(grid_dim, K, K)``.
        """
        dtype = Xb.dtype
        K = Xb.shape[1]
        I = jnp.identity(K, dtype=dtype)
        U = jnp.ones((K, K), dtype=dtype) / K
        Yb_pert = Yb @ (I - U)
        rinv = rinv_diag.astype(dtype)
        n_obs = int(rinv.shape[0])

        if self._use_patch:
            # Sparse gather: each lane reads only its <= P nearest obs (§14.3).
            patch_idx, patch_w = self._build_patch_w(n_obs, dtype)

            def _lane_A(args):
                idx_g, w_g = args                              # (Pw,), (Pw,)
                Yb_g = Yb_pert[idx_g]                          # (Pw, K) gather
                rinv_g = rinv[idx_g] * w_g                     # (Pw,) rinv*taper
                YtRinv = Yb_g.T * rinv_g[None, :]              # (K, Pw)
                return (K - 1) / rho * I + YtRinv @ Yb_g       # (K, K) SPD

            lane_inputs = (patch_idx, patch_w)
            G = patch_idx.shape[0]
        else:
            taper = self._build_taper(obs_loc_indices, dtype)

            def _lane_A(taper_row):
                rinv_local = rinv * taper_row.astype(dtype)    # (n_obs,)
                YtRinv = Yb_pert.T * rinv_local[None, :]       # (K, n_obs)
                return (K - 1) / rho * I + YtRinv @ Yb_pert    # (K, K) SPD

            lane_inputs = taper
            G = taper.shape[0]

        # Chunk over grid points EXACTLY as ``_local_columns`` (via
        # ``jax.lax.map`` in blocks of ``self.grid_chunk``) so the peak
        # ``(chunk, K, .)`` intermediate stays bounded -- a whole-grid
        # ``vmap`` at the real T42 grid OOMs the GPU on the dense path.
        chunk = self.grid_chunk
        if chunk is None or chunk >= G:
            return jax.vmap(_lane_A)(lane_inputs)              # (grid_dim, K, K)
        n_chunks = -(-G // chunk)                              # ceil div
        pad = n_chunks * chunk - G

        def _pad_reshape(arr):
            arr_p = jnp.pad(arr, ((0, pad),) + ((0, 0),) * (arr.ndim - 1))
            return arr_p.reshape((n_chunks, chunk) + arr.shape[1:])

        lane_c = jax.tree_util.tree_map(_pad_reshape, lane_inputs)
        A_c = jax.lax.map(lambda blk: jax.vmap(_lane_A)(blk), lane_c)
        return A_c.reshape(n_chunks * chunk, K, K)[:G]         # (grid_dim, K, K)

    # ── local (per-gridpoint) ETKF solve ──────────────────────────────────
    def _local_columns(self,
                       Xb_grid: ArrayLike,
                       Yb: ArrayLike,
                       Y: ArrayLike,
                       rinv_diag: ArrayLike,
                       taper: ArrayLike,
                       rho: float) -> jax.Array:
        """Fused per-gridpoint local ETKF: returns the analysis grid columns.

        vmaps over grid points; each lane tapers the FULL window-stacked
        ``rinv_diag`` by that gridpoint's Gaspari-Cohn row, solves the K x K
        ETKF transform (eigh-based symmetric sqrt), and recombines its OWN
        background column only -- returning ``(grid_dim, K)`` and never
        materializing the per-gridpoint ``Wa``.

        The grid is processed in blocks of ``self.grid_chunk`` (via
        :func:`jax.lax.map`) so the peak ``(chunk, K, n_obs)`` intermediate
        stays bounded; the result matches a whole-grid ``vmap`` to round-off
        (same lane / same order; ``lax.map`` only reorders XLA fusion).
        ``grid_chunk=None`` runs a single block (legacy whole-grid ``vmap``).

        Args:
            Xb_grid: Background ensemble in grid space, ``(grid_dim, K)``.
            Yb: Obs-space ensemble, ``(n_obs, K)``.
            Y: Flattened observation vector, ``(n_obs,)``.
            rinv_diag: Masked diagonal ``R^{-1}``, ``(n_obs,)``.
            taper: Gaspari-Cohn taper, ``(grid_dim, n_obs)``.
            rho: Multiplicative inflation factor.

        Returns:
            Analysis perturbation-and-mean grid columns, ``(grid_dim, K)``.
        """
        K = Yb.shape[1]
        dtype = Xb_grid.dtype
        I = jnp.identity(K, dtype=dtype)
        U = jnp.ones((K, K), dtype=dtype) / K
        yb_bar = jnp.mean(Yb, axis=1)
        Yb_pert = Yb @ (I - U)
        innov = (Y - yb_bar).astype(dtype)
        rinv = rinv_diag.astype(dtype)
        # SPD solver for the K x K transform ``A``: the shared ``_solve_pa_wa``
        # helper (also used by ETKF/ETKF4D) gives ``Pa = A^{-1}`` and
        # ``Wa = ((K-1) A^{-1})^{1/2}`` via eigh (``None``/``qr``/``jacobi``) or
        # the eigh-FREE Newton-Schulz path.  Its per-lane NS residual is not
        # surfaced here (the vmap returns only the analysis columns); NS
        # convergence is reported by the single-solve ETKF/ETKF4D path.
        eigh_impl = self.eigh_impl
        ns_iters = self.ns_iters
        n_obs = int(rinv.shape[0])

        if self._use_patch:
            # Sparse gather (§14.3): the lane reads only its <= P nearest obs.
            # ``rinv`` already carries per-cycle validity (zeroed for inactive
            # obs), so ``rinv[idx] * patch_w`` matches the dense ``rinv*taper``;
            # an all-zero patch_w -> A=(K-1)/rho I -> analysis = background.
            patch_idx, patch_w = self._build_patch_w(n_obs, dtype)

            def _lane(xb_col, idx_g, w_g):
                Yb_g = Yb_pert[idx_g]                          # (Pw, K) gather
                rinv_g = rinv[idx_g] * w_g                     # (Pw,) rinv*taper
                YtRinv = Yb_g.T * rinv_g[None, :]              # (K, Pw)
                A = (K - 1) / rho * I + YtRinv @ Yb_g
                innov_g = innov[idx_g]                         # (Pw,) gather
                Pa, Wa, _ = _solve_pa_wa(A, eigh_impl, ns_iters)
                wa = Pa @ (YtRinv @ innov_g)                   # (K,)
                xb_bar = jnp.mean(xb_col)
                xb_pert = xb_col - xb_bar                      # (K,)
                xa_pert = xb_pert @ Wa                         # (K,)
                return xa_pert + xb_bar + jnp.dot(xb_pert, wa)

            lane_extra = (patch_idx, patch_w)
        else:
            def _lane(xb_col, taper_row):
                rinv_local = rinv * taper_row.astype(dtype)    # (n_obs,)
                YtRinv = Yb_pert.T * rinv_local[None, :]       # (K, n_obs)
                A = (K - 1) / rho * I + YtRinv @ Yb_pert
                Pa, Wa, _ = _solve_pa_wa(A, eigh_impl, ns_iters)
                wa = Pa @ (YtRinv @ innov)                     # (K,)
                xb_bar = jnp.mean(xb_col)
                xb_pert = xb_col - xb_bar                      # (K,)
                xa_pert = xb_pert @ Wa                         # (K,)
                return xa_pert + xb_bar + jnp.dot(xb_pert, wa)

            lane_extra = (taper,)

        G = Xb_grid.shape[0]
        chunk = self.grid_chunk
        if chunk is None or chunk >= G:
            return jax.vmap(_lane)(Xb_grid, *lane_extra)       # (grid_dim, K)

        # Pad the grid up to a whole number of chunks, reshape to
        # (n_chunks, chunk, ...), map vmap(_lane) over the chunk axis (peak =
        # one block), then flatten and drop the padding rows.  Padding rows are
        # zeros and are sliced off, so they never affect the real output.
        n_chunks = -(-G // chunk)                              # ceil div
        pad = n_chunks * chunk - G

        def _pad_reshape(arr):
            arr_p = jnp.pad(arr, ((0, pad),) + ((0, 0),) * (arr.ndim - 1))
            return arr_p.reshape((n_chunks, chunk) + arr.shape[1:])

        Xb_c = _pad_reshape(Xb_grid)
        extra_c = jax.tree_util.tree_map(_pad_reshape, lane_extra)

        def _block(args):
            xb_blk, extra_blk = args
            return jax.vmap(_lane)(xb_blk, *extra_blk)         # (chunk, K)

        Xa_c = jax.lax.map(_block, (Xb_c, extra_c))            # (n_chunks,chunk,K)
        return Xa_c.reshape(n_chunks * chunk, K)[:G]           # (grid_dim, K)

    def _localized_analysis(self,
                            Xb: ArrayLike,
                            Yb: ArrayLike,
                            Y: ArrayLike,
                            rinv_diag: ArrayLike,
                            obs_loc_indices: ArrayLike,
                            rho: float,
                            key: ArrayLike | None = None) -> ArrayLike:
        """Full LETKF analysis: lift -> local solves -> project -> relax.

        Args:
            Xb: Background ensemble in spectral/state space, ``(system_dim, K)``.
            Yb: Obs-space ensemble, ``(n_obs, K)``.
            Y: Flattened observation vector, ``(n_obs,)``.
            rinv_diag: Masked diagonal ``R^{-1}``, ``(n_obs,)``.
            obs_loc_indices: Flattened observed grid indices (for the taper).
            rho: Multiplicative inflation factor.
            key: Optional per-cycle PRNG key enabling structured additive
                inflation on the spectral perturbations; ``None`` skips it.

        Returns:
            Analysis ensemble in spectral/state space, ``(system_dim, K)``.
        """
        dtype = Xb.dtype
        K = Xb.shape[1]
        I = jnp.identity(K, dtype=dtype)
        U = jnp.ones((K, K), dtype=dtype) / K

        # 1. ISHT lift every member to grid space.
        Xb_grid = jax.vmap(self.to_grid, in_axes=1, out_axes=1)(Xb)
        # Patch mode gathers per-lane inside ``_local_columns`` (no dense taper);
        # only build the dense ``(grid_dim, n_obs)`` taper on the dense path.
        taper = None if self._use_patch else self._build_taper(
            obs_loc_indices, dtype)

        # 2. Fused per-gridpoint local ETKF -> analysis grid columns.
        Xa_grid = self._local_columns(
            Xb_grid, Yb, Y, rinv_diag, taper, rho)             # (grid_dim, K)

        # 3. SHT project back to spectral; split mean + perturbations so the
        #    relaxation/inflation acts on the SPECTRAL perturbations (RTPS is a
        #    nonlinear per-coord rescale that does NOT commute with the SHT).
        Xa_spec = jax.vmap(self.from_grid, in_axes=1, out_axes=1)(Xa_grid)

        # Spectral taper on the ANALYSIS INCREMENT (analysis - background),
        # per mode: ``Xa = Xb + T(ell) * (Xa - Xb)``.  Preserves the background
        # exactly; attenuates only the increment's high-wavenumber content so it
        # injects no small-scale energy above the taper passband.  Applied here
        # (before the mean/pert split) so relaxation acts on the tapered
        # analysis.  ``None`` -> no-op (byte-identical to prior callers).
        if self.increment_taper is not None:
            t = self.increment_taper.astype(dtype)[:, None]    # (system_dim,1)
            Xa_spec = Xb + t * (Xa_spec - Xb)
        Xb_bar = jnp.mean(Xb, axis=1)
        Xb_pert = Xb @ (I - U)
        Xa_bar = jnp.mean(Xa_spec, axis=1)
        Xa_pert = Xa_spec @ (I - U)

        # SHT-truncation + resolved-energy diagnostics (host-side floats).
        self._update_diagnostics(Xa_grid, Xa_spec, Xb)

        # 4. Relaxation / additive inflation on the spectral perturbations.
        Xa_pert = self._apply_rtps(Xb_pert, Xa_pert)
        Xa_pert = self._apply_rtpp(Xb_pert, Xa_pert)
        if key is not None:
            Xa_pert = self._apply_additive(Xb_pert, Xa_pert, key)
        v = jnp.ones((1, K), dtype=dtype)
        return Xa_pert + Xa_bar[:, None] @ v

    def _update_diagnostics(self, Xa_grid, Xa_spec, Xb) -> None:
        """Store SHT-truncation power + resolved-energy pre/post (means).

        Only stores CONCRETE values (skips silently when called under a JAX
        trace, e.g. inside ``jax.lax.scan``, so no tracer leaks onto ``self``).
        Callers wanting per-cycle diagnostics invoke the analysis eagerly.
        """
        try:
            Xa_grid_bar = jnp.mean(Xa_grid, axis=1)
            round_trip = self.to_grid(self.from_grid(Xa_grid_bar))
            num = jnp.linalg.norm(Xa_grid_bar - round_trip)
            den = jnp.linalg.norm(Xa_grid_bar)
            self.trunc_power = float(num / jnp.where(den > 0, den, 1.0))
            self.energy_pre = float(jnp.sum(jnp.mean(Xb, axis=1) ** 2))
            self.energy_post = float(jnp.sum(jnp.mean(Xa_spec, axis=1) ** 2))
        except (jax.errors.TracerArrayConversionError,
                jax.errors.ConcretizationTypeError):
            pass

    def _fgat_analysis(self,
                       Xb_tau: ArrayLike,
                       Yb: ArrayLike,
                       Y_eff: ArrayLike,
                       rinv_diag: ArrayLike,
                       obs_loc_flat: ArrayLike,
                       rho: float,
                       key: ArrayLike | None = None) -> ArrayLike:
        """Localized strict 3D-FGAT analysis at the analysis time ``tau``.

        The domain-localized counterpart of :meth:`ETKF._fgat_analysis`:
        delegates to :meth:`_localized_analysis` with the tau-time obs-space
        ensemble ``Yb`` and effective obs vector ``Y_eff`` (so the local
        innovation is the FGAT innovation), tapering by distance from
        ``obs_loc_flat``.
        """
        return self._localized_analysis(
                Xb_tau, Yb, Y_eff, rinv_diag, obs_loc_flat, rho, key=key)

    def _compute_analysis(self,
                          Xb: ArrayLike,
                          Y: ArrayLike,
                          H: ArrayLike | None,
                          h: Callable | None,
                          R: ArrayLike,
                          rho: float = 1.0) -> ArrayLike:
        """LETKF analysis (overrides :meth:`ETKF._compute_analysis`).

        Builds the obs-space ensemble and diagonal ``R^{-1}`` from the
        masked-``H`` innovation (matching the base ETKF obs plumbing), then
        delegates to :meth:`_localized_analysis`.  ``obs_loc_indices`` for
        the taper is recovered from the nonzero column of each ``H`` row.
        """
        Yb = self._apply_obsop(Xb, H, h)                       # (n_obs, K)
        Y = jnp.asarray(Y).reshape(-1)
        # Diagonal R^{-1}: R is the (masked) obs error covariance from the base
        # obsop; masked-out rows are all-zero in H so their innovation is 0.
        rinv_diag = jnp.where(jnp.diag(R) > 0, 1.0 / jnp.diag(R), 0.0)
        obs_loc_indices = jnp.argmax(jnp.abs(H), axis=1)
        return self._localized_analysis(
            Xb, Yb, Y, rinv_diag, obs_loc_indices, rho)
