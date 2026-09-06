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


def build_patch_geometry_series(grid_latlon: ArrayLike,
                                obs_latlon_series,
                                localize_radius: float,
                                localize_units: str = "km",
                                patch_size: int | None = None,
                                producer: bool = False,
                                ):
    """Per-cycle local-patch geometry for Regime-B LETKF localization.

    Regime B (§14.4) is the MOVING-observer case: the obs positions differ
    every cycle, so the gridpoint->obs geometry is NOT static and must be
    rebuilt per cycle.  This is the B1 (host k-d tree per cycle) route from the
    design doc: given the KNOWN obs schedule, build the whole
    ``(n_cycles, grid_dim, P)`` geometry stack ONCE up front and consume it as a
    per-cycle scan input (the cycler indexes row ``t`` each cycle -- no host
    round-trip mid-scan).  Reuses :func:`build_patch_geometry` per cycle, so the
    metric and Gaspari-Cohn weights are IDENTICAL to Regime A; when every
    cycle's obs positions equal the static pool the two are byte-identical.

    A single GLOBAL ``P`` is used across all cycles (the JIT/scan shape
    contract): ``P = max`` over cycles of the exact per-cycle neighbour count
    (or the supplied ``patch_size``, with the usual logged truncation of the
    farthest obs on over-full cycles/grid points).

    Args:
        grid_latlon: ``(grid_dim, 2)`` (lat_deg, lon_deg) per grid point;
            stationary across cycles (the model grid does not move).
        obs_latlon_series: per-cycle obs positions -- either a length-``n_cycles``
            sequence of ``(n_obs_t, 2)`` arrays (ragged obs counts allowed) or a
            single ``(n_cycles, n_obs, 2)`` array.  Each entry is that cycle's
            obs slice (NOT window-stacked); 4D window-stacking is handled on
            device by :meth:`LETKF._build_patch_w`.
        localize_radius: Gaspari-Cohn half-width ``c`` (support cutoff ``2c``).
        localize_units: ``"km"`` (default) or ``"deg"``.
        patch_size: Fixed global ``P``.  ``None`` -> the exact max neighbour
            count over ALL cycles (no truncation).
        producer: If True return a :class:`PatchGeometryProducer` that builds
            cycle 0 synchronously and the rest on a background thread (the first
            cycle is available immediately; the remainder fill while the caller
            sets up the run).  If False (default) build the full stack eagerly.

    Returns:
        If ``producer`` is False: ``(patch_idx, patch_gc, diag)`` with
        ``patch_idx`` ``(n_cycles, grid_dim, P)`` int32, ``patch_gc``
        ``(n_cycles, grid_dim, P)`` float64, and ``diag`` a dict with the global
        ``P`` and per-cycle diagnostic lists.  If ``producer`` is True: a
        :class:`PatchGeometryProducer` (call ``.stack()`` for the same tuple).
    """
    series = _normalize_obs_series(obs_latlon_series)
    if producer:
        return PatchGeometryProducer(
            grid_latlon, series, localize_radius, localize_units, patch_size)
    return _build_series_eager(
        grid_latlon, series, localize_radius, localize_units, patch_size)


def _normalize_obs_series(obs_latlon_series) -> list:
    """Coerce the per-cycle obs positions into a list of ``(n_obs_t, 2)``."""
    if isinstance(obs_latlon_series, (list, tuple)):
        out = [np.asarray(o, dtype=np.float64) for o in obs_latlon_series]
    else:
        arr = np.asarray(obs_latlon_series, dtype=np.float64)
        if arr.ndim != 3 or arr.shape[2] != 2:
            raise ValueError(
                "obs_latlon_series array must be (n_cycles, n_obs, 2); got "
                f"{arr.shape}")
        out = [arr[t] for t in range(arr.shape[0])]
    if len(out) == 0:
        raise ValueError("obs_latlon_series must contain at least one cycle.")
    for t, o in enumerate(out):
        if o.ndim != 2 or o.shape[1] != 2:
            raise ValueError(
                f"obs_latlon_series[{t}] must be (n_obs_t, 2); got {o.shape}")
    return out


def _build_series_eager(grid_latlon, series, localize_radius, localize_units,
                        patch_size) -> tuple[np.ndarray, np.ndarray, dict]:
    """Build the full ``(n_cycles, grid_dim, P)`` stack, one cycle at a time.

    A single global ``P`` is chosen (max exact neighbour count across cycles
    when ``patch_size`` is None) so every cycle's ``(grid_dim, P)`` slab has the
    same shape; per-cycle geometries are then re-emitted at that ``P`` and
    stacked.  The per-cycle padding pool differs (each cycle references its OWN
    obs slice), which is recorded so :meth:`LETKF._build_patch_w` can window-
    stack correctly; here every cycle is a single (non-stacked) slice.
    """
    n_cycles = len(series)
    if patch_size is None:
        P = 1
        for o in series:
            _, _, d = build_patch_geometry(
                grid_latlon, o, localize_radius, localize_units, None)
            P = max(P, int(d["P"]))
    else:
        P = int(patch_size)
    idx_stack, gc_stack, per_cycle = [], [], []
    for o in series:
        pidx, pgc, d = build_patch_geometry(
            grid_latlon, o, localize_radius, localize_units, patch_size=P)
        idx_stack.append(pidx)
        gc_stack.append(pgc)
        per_cycle.append(d)
    patch_idx = np.stack(idx_stack, axis=0).astype(np.int32)
    patch_gc = np.stack(gc_stack, axis=0).astype(np.float64)
    diag = {
        "P": int(P),
        "n_cycles": int(n_cycles),
        "pool_sizes": [int(o.shape[0]) for o in series],
        "patch_max": [int(d["patch_max"]) for d in per_cycle],
        "pct_truncated": [float(d["pct_truncated"]) for d in per_cycle],
        "weight_mass_discarded": [float(d["weight_mass_discarded"])
                                  for d in per_cycle],
    }
    return patch_idx, patch_gc, diag


class PatchGeometryProducer:
    """Async host-side builder for the Regime-B per-cycle patch geometry stack.

    The DA cycle loop is sequential: it needs cycle 0's ``(grid_dim, P)``
    geometry immediately, but the remaining cycles' geometries can be built on a
    background thread while the caller finishes run setup (and, since the whole
    ``lax.scan`` consumes the completed stack, before the scan launches).  This
    matches the intent "the first timestep is needed immediately, then fill in
    while the rest is prepared".

    ``build0()`` returns cycle 0's slab synchronously and kicks off the worker;
    ``stack()`` blocks until the full ``(n_cycles, grid_dim, P)`` stack is ready
    and returns the same ``(patch_idx, patch_gc, diag)`` as the eager builder.
    A single global ``P`` (computed up front) fixes the slab shape so the worker
    can fill a preallocated stack in place.
    """

    def __init__(self, grid_latlon, series, localize_radius, localize_units,
                 patch_size):
        import threading
        self._grid = np.asarray(grid_latlon, dtype=np.float64)
        self._series = series
        self._radius = float(localize_radius)
        self._units = str(localize_units)
        self._n = len(series)
        self._G = int(self._grid.shape[0])
        # Fix the global P up front (needs one ball-count pass per cycle when
        # patch_size is None) so every slab shares a shape.
        if patch_size is None:
            P = 1
            for o in series:
                _, _, d = build_patch_geometry(
                    self._grid, o, self._radius, self._units, None)
                P = max(P, int(d["P"]))
        else:
            P = int(patch_size)
        self._P = P
        self._idx = np.zeros((self._n, self._G, P), dtype=np.int32)
        self._gc = np.zeros((self._n, self._G, P), dtype=np.float64)
        self._per_cycle = [None] * self._n
        self._done = threading.Event()
        self._built0 = False
        self._thread = None

    def _build_one(self, t: int) -> None:
        pidx, pgc, d = build_patch_geometry(
            self._grid, self._series[t], self._radius, self._units,
            patch_size=self._P)
        self._idx[t] = pidx
        self._gc[t] = pgc
        self._per_cycle[t] = d

    def build0(self) -> tuple[np.ndarray, np.ndarray, dict]:
        """Build + return cycle 0's slab now; start filling the rest async."""
        import threading
        if not self._built0:
            self._build_one(0)
            self._built0 = True

            def _worker():
                for t in range(1, self._n):
                    self._build_one(t)
                self._done.set()

            if self._n == 1:
                self._done.set()
            else:
                self._thread = threading.Thread(target=_worker, daemon=True)
                self._thread.start()
        return self._idx[0], self._gc[0], (self._per_cycle[0] or {})

    def stack(self) -> tuple[np.ndarray, np.ndarray, dict]:
        """Block until the full stack is built; return (idx, gc, diag)."""
        if not self._built0:
            self.build0()
        self._done.wait()
        per = self._per_cycle
        diag = {
            "P": int(self._P),
            "n_cycles": int(self._n),
            "pool_sizes": [int(o.shape[0]) for o in self._series],
            "patch_max": [int(d["patch_max"]) for d in per],
            "pct_truncated": [float(d["pct_truncated"]) for d in per],
            "weight_mass_discarded": [float(d["weight_mass_discarded"])
                                      for d in per],
        }
        return self._idx, self._gc, diag


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
                 patch_idx_series: ArrayLike | None = None,
                 patch_gc_series: ArrayLike | None = None,
                 patch_pool_sizes: ArrayLike | None = None,
                 patch_callback: Callable | None = None,
                 patch_callback_P: int | None = None,
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
        # Regime-B per-cycle local-patch geometry (§14.4, B1): a STACKED
        # ``(n_cycles, grid_dim, P)`` geometry from
        # :func:`build_patch_geometry_series` for MOVING observers.  Each cycle
        # ``t`` indexes row ``t`` (derived on device from ``cur_time``), so the
        # gather lane is the same as Regime A but the geometry changes per cycle.
        # Consumed as a device-resident stack (built host-side up front from the
        # known obs schedule) -- NOT rebuilt inside ``lax.scan``.  Both-or-neither
        # with ``patch_gc_series``; mutually exclusive with the static Regime-A
        # ``patch_idx``.  ``patch_pool_sizes`` is the per-cycle single-slice obs
        # count (for window-stacking); if omitted it is inferred as the last axis
        # is a single slice (n_times inferred at solve time from the obs axis).
        if (patch_idx_series is None) != (patch_gc_series is None):
            raise ValueError(
                "patch_idx_series and patch_gc_series must be supplied together "
                "(or both None).")
        if patch_idx_series is not None and patch_idx is not None:
            raise ValueError(
                "supply EITHER the static Regime-A patch_idx/patch_gc OR the "
                "per-cycle Regime-B patch_idx_series/patch_gc_series, not both.")
        self._patch_idx_series = (
            None if patch_idx_series is None
            else jnp.asarray(patch_idx_series, dtype=jnp.int32))
        self._patch_gc_series = (None if patch_gc_series is None
                                 else jnp.asarray(patch_gc_series))
        self._patch_pool_sizes = None
        self._patch_pool_host = None
        if self._patch_idx_series is not None:
            if self._patch_idx_series.ndim != 3:
                raise ValueError(
                    "patch_idx_series must be (n_cycles, grid_dim, P); got "
                    f"{self._patch_idx_series.shape}")
            if self._patch_idx_series.shape != self._patch_gc_series.shape:
                raise ValueError(
                    f"patch_idx_series {self._patch_idx_series.shape} and "
                    f"patch_gc_series {self._patch_gc_series.shape} must have "
                    "the same (n_cycles, grid_dim, P) shape.")
            n_cyc = int(self._patch_idx_series.shape[0])
            if patch_pool_sizes is None:
                # No per-cycle pool sizes given: infer a single shared pool from
                # obs_latlon (the common stationary-count case is Regime A;
                # for B the caller should pass patch_pool_sizes).  Fall back to
                # max index + 1 per cycle.
                if self._obs_latlon is not None:
                    self._patch_pool_sizes = jnp.full(
                        (n_cyc,), int(self._obs_latlon.shape[0]),
                        dtype=jnp.int32)
                else:
                    self._patch_pool_sizes = (
                        self._patch_idx_series.max(axis=(1, 2)) + 1
                        ).astype(jnp.int32)
            else:
                self._patch_pool_sizes = jnp.asarray(
                    patch_pool_sizes, dtype=jnp.int32).reshape(-1)
                if int(self._patch_pool_sizes.shape[0]) != n_cyc:
                    raise ValueError(
                        f"patch_pool_sizes length {self._patch_pool_sizes.shape[0]}"
                        f" != n_cycles {n_cyc}.")
            # Host-known constant single-slice pool size for window-stacking:
            # ``n_times = n_obs // pool`` is a STATIC shape, so the pool count
            # must be a Python int and constant across the run.  Derive it from
            # the host-side pool sizes and require they are all equal.
            _pools_host = np.asarray(
                patch_pool_sizes if patch_pool_sizes is not None
                else np.asarray(self._patch_pool_sizes)).reshape(-1)
            if _pools_host.size and not np.all(_pools_host == _pools_host[0]):
                raise ValueError(
                    "Regime-B window-stacking requires a CONSTANT per-cycle obs "
                    "slice count across the run (n_times is a static shape); "
                    f"got varying patch_pool_sizes={_pools_host.tolist()}.")
            self._patch_pool_host = (int(_pools_host[0]) if _pools_host.size
                                     else None)
        # Regime-B FALLBACK for an UNKNOWN obs schedule (§14.4): rebuild the
        # ``(grid_dim, P)`` geometry on the HOST every cycle via
        # :func:`jax.pure_callback`, from that cycle's live obs positions
        # (``grid_latlon[obs_loc_indices]``).  This is the general moving-obs
        # route when the schedule cannot be precomputed; it SERIALIZES the GPU
        # cycle (a host round-trip per analysis) and is slower than the
        # precomputed stack -- prefer the stack when the schedule is known.
        # ``patch_callback`` is ``(grid_ll, obs_ll, radius, units, P) ->
        # (patch_idx (grid_dim,P) int32, patch_gc (grid_dim,P) float)``; default
        # uses :func:`build_patch_geometry`.  ``patch_callback_P`` fixes ``P``
        # (the static output shape) and is REQUIRED.  Mutually exclusive with the
        # static Regime-A patch_idx and the Regime-B series.
        self._patch_callback = patch_callback
        self._patch_callback_P = (None if patch_callback_P is None
                                  else int(patch_callback_P))
        if patch_callback is not None:
            if patch_idx is not None or patch_idx_series is not None:
                raise ValueError(
                    "patch_callback is mutually exclusive with patch_idx "
                    "(Regime A) and patch_idx_series (precomputed Regime B).")
            if self._patch_callback_P is None or self._patch_callback_P < 1:
                raise ValueError(
                    "patch_callback requires patch_callback_P >= 1 (the fixed "
                    "patch size P setting the static geometry shape).")
            if self._grid_latlon is None:
                raise ValueError(
                    "patch_callback requires grid_latlon (the moving-obs "
                    "geometry is built from grid_latlon[obs_loc_indices]).")
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

    def _distances(self, obs_loc_indices: ArrayLike,
                   grid_rows: slice | None = None) -> jax.Array:
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

        ``grid_rows`` (optional) restricts the GRID-point axis to a slice so
        only ``(len(rows), n_obs)`` is formed -- used by the block-streamed
        capture path so the full ``(grid_dim, n_obs)`` distance/taper matrix
        (~700 MiB at the real T42 grid) is never materialized on the GPU.
        The obs axis and every distance value are byte-identical to the full
        matrix's corresponding rows (same metric, same order).
        """
        obs_idx = jnp.asarray(obs_loc_indices).reshape(-1).astype(jnp.int32)
        n_obs = int(obs_idx.shape[0])
        if self._grid_latlon is not None:
            grid_ll = self._grid_latlon                       # (grid_dim, 2)
            if grid_rows is not None:
                grid_ll = grid_ll[grid_rows]
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
        if grid_rows is not None:
            grid_pos = grid_pos[grid_rows]
        obs_pos = obs_idx.astype(jnp.float32)[None, :]
        d = jnp.abs(grid_pos - obs_pos)
        return jnp.minimum(d, n - d)

    def _build_taper(self, obs_loc_indices: ArrayLike,
                     dtype, grid_rows: slice | None = None) -> jax.Array:
        """Dense ``(grid_dim, n_obs)`` Gaspari-Cohn taper matrix.

        The obs network is a fixed pool, so the gridpoint->obs geometry is
        static and the taper is cached across cycles.  The cache is only
        populated with a CONCRETE array: when this is first reached under a
        JAX trace (``jax.lax.scan`` in :meth:`DACycler.cycle`) the taper is
        recomputed each iteration rather than caching a tracer -- caching a
        tracer would let it escape the trace and raise
        :class:`jax.errors.UnexpectedTracerError` on the next cycle, making
        the cycler single-use.  Exactly 0 beyond ``2c``.

        ``grid_rows`` (optional) restricts the GRID-point axis to a slice so
        only that block of taper rows is formed -- the block-streamed capture
        path passes it so the full ``(grid_dim, n_obs)`` taper is never built
        on-device.  A partial slice bypasses the whole-grid cache on BOTH read
        and write (the cache holds the full matrix only), so the returned rows
        are byte-identical to the corresponding rows of the full taper.
        """
        if grid_rows is None and self._taper_cache is not None:
            return self._taper_cache.astype(dtype)
        dist = self._distances(obs_loc_indices, grid_rows=grid_rows)
        taper = _gaspari_cohn(dist, self._radius_km())
        if (grid_rows is None
                and not isinstance(jnp.asarray(obs_loc_indices), jax.core.Tracer)):
            self._taper_cache = taper
        return taper.astype(dtype)

    @property
    def _use_patch(self) -> bool:
        """Whether ANY sparse local-patch gather (Regime A or B) is active."""
        return (self._patch_idx is not None
                or self._patch_idx_series is not None
                or self._patch_callback is not None)

    @property
    def _use_patch_series(self) -> bool:
        """Whether the precomputed per-cycle Regime-B geometry stack is active."""
        return self._patch_idx_series is not None

    @property
    def _use_patch_callback(self) -> bool:
        """Whether the per-cycle host-callback Regime-B fallback is active."""
        return self._patch_callback is not None

    def _select_patch(self, cycle_idx):
        """Return this cycle's static ``(grid_dim, P)`` geometry + pool size.

        Regime A: the single static ``(patch_idx, patch_gc, pool)`` held on
        ``self`` (``cycle_idx`` ignored).  Regime B: index row ``cycle_idx`` of
        the ``(n_cycles, grid_dim, P)`` stack -- a device gather, so this is
        ``lax.scan``-safe (``cycle_idx`` may be a tracer).  ``pool`` is the
        single-slice obs count that :meth:`_build_patch_w` uses to window-stack.
        """
        if self._patch_idx_series is not None:
            t = (0 if cycle_idx is None
                 else jnp.asarray(cycle_idx).astype(jnp.int32))
            n_cyc = int(self._patch_idx_series.shape[0])
            t = jnp.clip(t, 0, n_cyc - 1)
            p_idx = self._patch_idx_series[t]                  # (grid_dim, P)
            p_gc = self._patch_gc_series[t]                    # (grid_dim, P)
            pool = self._patch_pool_sizes[t]                   # scalar (int32)
            return p_idx, p_gc, pool
        return self._patch_idx, self._patch_gc, self._patch_pool_size

    def _callback_patch(self, obs_latlon_t):
        """Host-rebuild this cycle's ``(grid_dim, P)`` geometry via pure_callback.

        The Regime-B fallback for an unknown obs schedule: given this cycle's
        obs positions ``obs_latlon_t (pool, 2)``, call the (default or supplied)
        host builder on the CPU each cycle and return the fixed-``P`` geometry.
        Uses :func:`jax.pure_callback` so it is ``lax.scan``-legal (at the cost
        of a host round-trip that serializes the GPU cycle).
        """
        import functools
        grid_ll = np.asarray(self._grid_latlon, dtype=np.float64)
        P = int(self._patch_callback_P)
        G = int(grid_ll.shape[0])
        radius = float(self.localize_radius)
        units = str(self.localize_units)
        builder = self._patch_callback

        def _host(obs_ll):
            obs = np.asarray(obs_ll, dtype=np.float64)
            pidx, pgc = builder(grid_ll, obs, radius, units, P)
            return (np.asarray(pidx, dtype=np.int32),
                    np.asarray(pgc, dtype=np.float64))

        out_shapes = (jax.ShapeDtypeStruct((G, P), jnp.int32),
                      jax.ShapeDtypeStruct((G, P), jnp.float64))
        p_idx, p_gc = jax.pure_callback(_host, out_shapes, obs_latlon_t)
        return p_idx, p_gc

    def _build_patch_w(self, n_obs: int, dtype, cycle_idx=None,
                       obs_latlon_t=None) -> tuple[jax.Array, jax.Array]:
        """Window-stacked patch gather indices + weights for the local solve.

        Expands this cycle's single-slice geometry ``(grid_dim, P)`` onto the
        (possibly window-stacked) obs axis of length ``n_obs``.  In 4D the obs
        axis repeats the pool ``n_times`` times (``n_obs = n_times * pool``), so
        a grid point's patch in window block ``t`` gathers pool slots
        ``patch_idx + t * pool``; the DISTANCE-ONLY Gaspari-Cohn weight is
        identical in every block (same geometry within a cycle, exactly as the
        dense taper tiles).  Per-cycle observation ACTIVITY is NOT applied here:
        it is carried by the zeroed ``rinv_diag`` entries the lane gathers
        (``rinv[gathered_idx]``), matching the dense path's ``rinv * taper``.

        Three geometry sources: Regime A (static, cached across cycles);
        precomputed Regime-B series (indexed by ``cycle_idx``); and the
        Regime-B host callback (``obs_latlon_t`` -> :meth:`_callback_patch`,
        rebuilt each cycle).  For the callback the single-slice pool size is the
        callback obs count (``obs_latlon_t`` rows) which must match
        ``patch_callback_P``'s pool; ``n_obs`` must be a static Python int (the
        obs axis length is a shape).

        Returns ``(patch_idx_ws, patch_w_ws)``, both ``(grid_dim, n_times * P)``:
        int32 obs-axis indices and the dtype-cast GC weights.
        """
        callback = self._use_patch_callback
        series = self._use_patch_series
        cacheable = not series and not callback
        if cacheable and self._patch_w_cache is not None:
            idx_c, w_c = self._patch_w_cache
            if int(idx_c.shape[1]) == n_obs // self._patch_pool_size \
                    * self._patch_idx.shape[1]:
                return idx_c, w_c.astype(dtype)
            # Obs-axis length changed (different window stacking) -> rebuild.
            self._patch_w_cache = None
        if callback:
            if obs_latlon_t is None:
                raise ValueError(
                    "patch_callback mode requires obs_latlon_t (this cycle's "
                    "obs positions) to be threaded into the local solve.")
            p_idx, p_gc = self._callback_patch(obs_latlon_t)
            pool = int(jnp.asarray(obs_latlon_t).shape[0])
        else:
            p_idx, p_gc, pool = self._select_patch(cycle_idx)
        if callback:
            if pool <= 0 or n_obs % pool != 0:
                raise ValueError(
                    f"obs axis ({n_obs}) is not an integer multiple of the "
                    f"callback pool size ({pool}).")
            n_times = n_obs // pool
            pool_off = pool
        elif not series:
            if pool is None or pool <= 0:
                raise ValueError(
                    "patch localization requires a known pool size; supply "
                    "obs_latlon or non-empty patch_idx.")
            if n_obs % int(pool) != 0:
                raise ValueError(
                    f"obs axis ({n_obs}) is not an integer multiple of the patch"
                    f" pool size ({pool}); cannot align the local patches to the"
                    " window-stacked observations.")
            n_times = n_obs // int(pool)
            pool_off = int(pool)
        else:
            # Regime B: pool is a traced scalar.  n_times is a STATIC int
            # (n_obs // per-cycle pool); every cycle in the series shares the
            # same P and (validated at construction) the same pool count, so use
            # the host-known constant pool size to keep shapes static.
            pool_host = self._patch_pool_host
            if pool_host is None or pool_host <= 0 or n_obs % pool_host != 0:
                raise ValueError(
                    f"obs axis ({n_obs}) is not an integer multiple of the "
                    f"Regime-B pool size ({pool_host}); the per-cycle obs slice "
                    "count must be constant across the run for window-stacking.")
            n_times = n_obs // pool_host
            pool_off = pool_host             # host int (pool constant across run)
        P = p_idx.shape[1]
        # (grid_dim, P) -> (grid_dim, n_times, P) with per-block pool offset,
        # then flatten the (n_times, P) axes to (grid_dim, n_times*P).
        base = p_idx[:, None, :]                               # (G, 1, P)
        offs = (jnp.arange(n_times, dtype=jnp.int32) * pool_off
                )[None, :, None]                               # (1, nt, 1)
        idx_ws = (base + offs).reshape(p_idx.shape[0], n_times * P)
        w_ws = jnp.broadcast_to(
            p_gc[:, None, :], (p_gc.shape[0], n_times, P)
            ).reshape(idx_ws.shape)
        if cacheable:
            self._patch_w_cache = (idx_ws, w_ws)
        return idx_ws, w_ws.astype(dtype)

    # ── local (per-gridpoint) ETKF transform capture ──────────────────────
    def capture_A_matrices(self,
                           Xb: ArrayLike,
                           Yb: ArrayLike,
                           Y: ArrayLike,
                           rinv_diag: ArrayLike,
                           obs_loc_indices: ArrayLike,
                           rho: float,
                           cycle_idx=None,
                           obs_latlon_t=None,
                           to_host: bool = False):
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
            cycle_idx: Regime-B (series) per-cycle geometry row selector
                (ignored in Regime A / dense).
            obs_latlon_t: Regime-B (callback) this cycle's obs positions
                ``(pool, 2)`` (ignored otherwise).
            to_host: If ``True``, transfer each grid block to host as it is
                computed and return a single concatenated ``np.ndarray`` (never
                materializing the whole ``(grid_dim, K, K)`` stack on-device on
                top of the resident DA working set -- this is what OOMs the
                capture path on a 24 GB GPU at the real T42 grid).  Requires a
                finite ``grid_chunk``.  Default ``False`` returns the on-device
                ``jax.Array`` (unchanged behaviour).

        Returns:
            The per-gridpoint SPD transform stack ``(grid_dim, K, K)`` as a
            ``jax.Array`` (``to_host=False``) or ``np.ndarray``
            (``to_host=True``).
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
            patch_idx, patch_w = self._build_patch_w(
                n_obs, dtype, cycle_idx, obs_latlon_t)

            def _lane_A(args):
                idx_g, w_g = args                              # (Pw,), (Pw,)
                Yb_g = Yb_pert[idx_g]                          # (Pw, K) gather
                rinv_g = rinv[idx_g] * w_g                     # (Pw,) rinv*taper
                YtRinv = Yb_g.T * rinv_g[None, :]              # (K, Pw)
                return (K - 1) / rho * I + YtRinv @ Yb_g       # (K, K) SPD

            lane_inputs = (patch_idx, patch_w)
            G = patch_idx.shape[0]
            W = int(patch_idx.shape[1])                        # patch width P
        else:
            def _lane_A(taper_row):
                rinv_local = rinv * taper_row.astype(dtype)    # (n_obs,)
                YtRinv = Yb_pert.T * rinv_local[None, :]       # (K, n_obs)
                return (K - 1) / rho * I + YtRinv @ Yb_pert    # (K, K) SPD

            # The dense taper is ``(grid_dim, n_obs)`` (~700 MiB at the real T42
            # grid); building it whole (even once) is itself the capture OOM, so
            # on the streaming path DEFER it and build only each block's rows
            # below.  Non-streaming callers still build it once here.
            G = int(self._grid_latlon.shape[0]
                    if self._grid_latlon is not None else self.system_dim)
            W = n_obs
            if not to_host:
                lane_inputs = self._build_taper(obs_loc_indices, dtype)

        # Chunk over grid points EXACTLY as ``_local_columns`` (via
        # ``jax.lax.map`` in blocks of ``self.grid_chunk``) so the peak
        # ``(chunk, K, .)`` intermediate stays bounded -- a whole-grid
        # ``vmap`` at the real T42 grid OOMs the GPU on the dense path.
        chunk = self.grid_chunk
        if chunk is None or chunk >= G:
            if self._use_patch or not to_host:
                A_full = jax.vmap(_lane_A)(lane_inputs)        # (grid_dim, K, K)
                return np.asarray(A_full) if to_host else A_full
            # dense + to_host + single block: build the whole taper once (the
            # caller opted out of chunking) and vmap.
            taper = self._build_taper(obs_loc_indices, dtype)
            return np.asarray(jax.vmap(_lane_A)(taper))

        if to_host:
            # Stream block-by-block to host: compute one ``(cap_chunk, K, K)``
            # block, copy it into the preallocated NumPy stack, and free the
            # device block before the next -- so the device never holds the
            # whole ``(grid_dim, K, K)`` stack on top of the resident DA
            # working set (the OOM that killed the L4 capture).
            #
            # The block size is bounded by the per-lane INTERMEDIATE, not by
            # ``self.grid_chunk``: vmapping ``_lane_A`` over a block of ``c``
            # grid points materializes a ``(c, K, W)`` ``Y^T R^{-1}`` temporary
            # (W = the window-stacked ``n_obs`` on the dense path, or the patch
            # width ``P`` on the sparse path) BEFORE the ``(c, K, K)`` output.
            # At the real T42 grid this dense temporary is ~2.8 GiB at c=256
            # (K=128, W~1.2e4, fp64) and OOMs the 24 GB L4 alongside the
            # resident DA working set -- so cap the block so that temporary
            # stays under ~256 MiB.  On the dense path the block's taper rows
            # are built INSIDE the loop (via ``grid_rows``) so the full
            # ``(grid_dim, n_obs)`` taper is never formed.  Smaller blocks are
            # byte-identical (same ``_lane_A``, same order; only the partition
            # changes -- see the ``grid_chunk``-varied equivalence test).
            budget_lanes = max(1, (256 * 1024 * 1024)
                               // (K * max(1, W) * np.dtype(dtype).itemsize))
            cap_chunk = max(1, min(chunk, budget_lanes))
            _lane_A_blk = jax.jit(jax.vmap(_lane_A))
            out = np.empty((G, K, K), dtype=np.dtype(dtype))
            for start in range(0, G, cap_chunk):
                stop = min(start + cap_chunk, G)
                if self._use_patch:
                    blk = jax.tree_util.tree_map(
                            lambda a, s=start, e=stop: a[s:e], lane_inputs)
                else:
                    blk = self._build_taper(
                            obs_loc_indices, dtype,
                            grid_rows=slice(start, stop))      # (<=cap_chunk,n_obs)
                A_blk = _lane_A_blk(blk)                       # (<=cap_chunk,K,K)
                out[start:stop] = np.asarray(A_blk)
                del A_blk
            return out

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
                       rho: float,
                       cycle_idx=None,
                       obs_latlon_t=None) -> jax.Array:
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
            cycle_idx: Regime-B (series) per-cycle geometry row selector
                (ignored in Regime A / dense).
            obs_latlon_t: Regime-B (callback) this cycle's obs positions
                ``(pool, 2)`` (ignored otherwise).

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
            patch_idx, patch_w = self._build_patch_w(
                n_obs, dtype, cycle_idx, obs_latlon_t)

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

        # jax.lax.map is scan-based: chunking alone only bounds the FORWARD
        # peak (one block materialized at a time). Without jax.checkpoint on
        # the per-block function, reverse-mode AD still retains every one of
        # the n_chunks blocks' activations for backward -- checkpointing here
        # recomputes one block from its own (xb_blk, extra_blk) input instead,
        # so backward peak is also bounded to ~one chunk. Safe: _block
        # operates on fixed-size padded chunks, no dynamic shapes.
        Xa_c = jax.lax.map(jax.checkpoint(_block), (Xb_c, extra_c))  # (n_chunks,chunk,K)
        return Xa_c.reshape(n_chunks * chunk, K)[:G]           # (grid_dim, K)

    def _localized_analysis(self,
                            Xb: ArrayLike,
                            Yb: ArrayLike,
                            Y: ArrayLike,
                            rinv_diag: ArrayLike,
                            obs_loc_indices: ArrayLike,
                            rho: float,
                            key: ArrayLike | None = None,
                            cycle_idx=None,
                            obs_latlon_t=None) -> ArrayLike:
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
            cycle_idx: Regime-B (series) per-cycle geometry row selector
                (``None`` in Regime A / dense; the 4D/FGAT call sites pass the
                scan cycle index derived from ``cur_time`` so the moving-obs
                geometry tracks the cycle).
            obs_latlon_t: Regime-B (callback) this cycle's obs positions
                ``(pool, 2)`` for the per-cycle host k-d tree rebuild
                (``None`` otherwise).

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
            Xb_grid, Yb, Y, rinv_diag, taper, rho, cycle_idx,
            obs_latlon_t)                                      # (grid_dim, K)

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

    def _cycle_index(self, cur_time):
        """Scan cycle index ``t`` from the carried ``cur_time`` (Regime B).

        Mirrors the additive-inflation key derivation
        (``round(cur_time / analysis_window)``); returns ``None`` when no
        per-cycle geometry stack is active so Regime A / dense pay nothing.
        Traced-safe (used only to index the device-resident geometry stack).
        """
        if not self._use_patch_series:
            return None
        return jnp.round(
            jnp.asarray(cur_time) / self.analysis_window).astype(jnp.int32)

    def _callback_obs_latlon_4d(self, cur_obs_loc_indices):
        """Single-slice obs positions ``(obs_dim, 2)`` for the Regime-B callback.

        Returns ``None`` unless the host-callback fallback is active.  The
        window-stacked location indices are ``(n_times, obs_dim)`` and the pool
        geometry recurs every window block (stationary within the window), so the
        FIRST block's positions ``grid_latlon[indices[0]]`` are the single-slice
        pool the callback rebuilds; :meth:`_build_patch_w` then window-stacks it.
        """
        if not self._use_patch_callback:
            return None
        idx2d = jnp.asarray(cur_obs_loc_indices)
        first = (idx2d[0] if idx2d.ndim == 2
                 else idx2d.reshape(-1)).astype(jnp.int32)
        return self._grid_latlon[first]                        # (obs_dim, 2)

    def _fgat_analysis(self,
                       Xb_tau: ArrayLike,
                       Yb: ArrayLike,
                       Y_eff: ArrayLike,
                       rinv_diag: ArrayLike,
                       obs_loc_flat: ArrayLike,
                       rho: float,
                       key: ArrayLike | None = None,
                       cycle_idx=None,
                       obs_latlon_t=None) -> ArrayLike:
        """Localized strict 3D-FGAT analysis at the analysis time ``tau``.

        The domain-localized counterpart of :meth:`ETKF._fgat_analysis`:
        delegates to :meth:`_localized_analysis` with the tau-time obs-space
        ensemble ``Yb`` and effective obs vector ``Y_eff`` (so the local
        innovation is the FGAT innovation), tapering by distance from
        ``obs_loc_flat``.
        """
        return self._localized_analysis(
                Xb_tau, Yb, Y_eff, rinv_diag, obs_loc_flat, rho, key=key,
                cycle_idx=cycle_idx, obs_latlon_t=obs_latlon_t)

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
        # Regime-B per-cycle geometry needs the scan cycle index, which the
        # legacy single-slice 3D path does not carry.  Regime B is a WINDOWED
        # (4D / 3D-FGAT) feature; require one of those cyclers rather than
        # silently reusing cycle-0 geometry every cycle.
        if self._use_patch_series or self._use_patch_callback:
            raise ValueError(
                "Regime-B per-cycle patch geometry (patch_idx_series / "
                "patch_callback) is only supported on the windowed cyclers "
                "(LETKF4D, or LETKF with fgat=True); the legacy single-slice 3D "
                "path has no per-cycle geometry hook.  Use the static Regime-A "
                "patch_idx for plain 3D.")
        Yb = self._apply_obsop(Xb, H, h)                       # (n_obs, K)
        Y = jnp.asarray(Y).reshape(-1)
        # Diagonal R^{-1}: R is the (masked) obs error covariance from the base
        # obsop; masked-out rows are all-zero in H so their innovation is 0.
        rinv_diag = jnp.where(jnp.diag(R) > 0, 1.0 / jnp.diag(R), 0.0)
        obs_loc_indices = jnp.argmax(jnp.abs(H), axis=1)
        return self._localized_analysis(
            Xb, Yb, Y, rinv_diag, obs_loc_indices, rho)
