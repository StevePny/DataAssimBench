"""Class for Local Ensemble Transform Kalman Filter (LETKF) DA Cycler."""

import numpy as np
import jax
import jax.numpy as jnp
import xarray as xr
from dabench import _xarray_jax as xj
from typing import Callable

from dabench.dacycler import ETKF


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
                 ns_iters: int = 20,
                 **kwargs):
        self.to_grid = (lambda x: x) if to_grid is None else to_grid
        self.from_grid = (lambda x: x) if from_grid is None else from_grid
        self._grid_latlon = (None if grid_latlon is None
                             else jnp.asarray(grid_latlon))
        self._obs_latlon = (None if obs_latlon is None
                            else jnp.asarray(obs_latlon))
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
        #               NS sidesteps eigh entirely.  See ``_spd_pa_wa_ns``.
        _valid = ("qr", "jacobi", "newton_schulz", "ns")
        if eigh_impl is not None and str(eigh_impl).lower() not in _valid:
            raise ValueError(
                "eigh_impl must be None, 'qr', 'jacobi', or 'newton_schulz', "
                f"got {eigh_impl!r}")
        _ei = None if eigh_impl is None else str(eigh_impl).lower()
        self.eigh_impl = "newton_schulz" if _ei == "ns" else _ei
        # Newton-Schulz iteration budget (fp64 well-conditioned SPD 64x64
        # reaches ~1e-12 in ~15 coupled steps; only used when NS is selected).
        self.ns_iters = int(ns_iters)
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
        # SPD solver for the K x K transform ``A``.  Two families, both giving
        # ``Pa = A^{-1}`` and ``Wa = ((K-1) A^{-1})^{1/2}`` for the SPD ``A``:
        #   * eigh-based (``None``/``qr``/``jacobi``): symmetric eigendecomp,
        #     ``Pa=(V/w)V^T``, ``Wa=(V sqrt((K-1)/w))V^T``.  ``None`` uses
        #     ``jnp.linalg.eigh`` (byte-identical to every prior caller); the
        #     lax path swaps its (v, w) return to jnp's (w, v).
        #   * ``newton_schulz``: eigh-FREE.  ``_spd_inv_sqrt_ns`` computes
        #     ``Z = A^{-1/2}`` via a matmul-only coupled iteration (batched GEMM
        #     on the accelerator), then ``Pa = Z Z``, ``Wa = sqrt(K-1) Z``.
        eigh_impl = self.eigh_impl
        if eigh_impl == "newton_schulz":
            _ns_iters = self.ns_iters
            def _pa_wa(A):
                Z = _spd_inv_sqrt_ns(A, n_iter=_ns_iters)      # A^{-1/2}
                Pa = Z @ Z                                     # A^{-1}
                Wa = jnp.sqrt(jnp.asarray(K - 1, dtype)) * Z   # ((K-1)A^{-1})^½
                return Pa, Wa
        else:
            if eigh_impl is None:
                _eigh = lambda A: jnp.linalg.eigh(A)
            else:
                # EighImplementation values are lowercase ("qr"/"jacobi");
                # the lax API returns (v, w) -- swap to jnp's (w, v).
                impl = jax.lax.linalg.EighImplementation(eigh_impl)
                _eigh = lambda A: (lambda v, w: (w, v))(
                    *jax.lax.linalg.eigh(A, implementation=impl))

            def _pa_wa(A):
                w_eig, V = _eigh(A)
                inv_eig = jnp.where(w_eig > 0, 1.0 / w_eig, 0.0)
                Pa = (V * inv_eig) @ V.T                       # pinv(A), SPD
                sqrt_eig = jnp.sqrt(jnp.clip((K - 1) * inv_eig, 0.0))
                Wa = (V * sqrt_eig) @ V.T                      # sqrtm((K-1)Pa)
                return Pa, Wa

        def _lane(xb_col, taper_row):
            rinv_local = rinv * taper_row.astype(dtype)        # (n_obs,)
            YtRinv = Yb_pert.T * rinv_local[None, :]           # (K, n_obs)
            A = (K - 1) / rho * I + YtRinv @ Yb_pert
            Pa, Wa = _pa_wa(A)
            wa = Pa @ (YtRinv @ innov)                         # (K,)
            xb_bar = jnp.mean(xb_col)
            xb_pert = xb_col - xb_bar                          # (K,)
            xa_pert = xb_pert @ Wa                             # (K,)
            xa = xa_pert + xb_bar + jnp.dot(xb_pert, wa)
            return xa

        G = Xb_grid.shape[0]
        chunk = self.grid_chunk
        if chunk is None or chunk >= G:
            return jax.vmap(_lane)(Xb_grid, taper)             # (grid_dim, K)

        # Pad the grid up to a whole number of chunks, reshape to
        # (n_chunks, chunk, ...), map vmap(_lane) over the chunk axis (peak =
        # one block), then flatten and drop the padding rows.  Padding rows are
        # zeros and are sliced off, so they never affect the real output.
        n_chunks = -(-G // chunk)                              # ceil div
        pad = n_chunks * chunk - G
        Xb_p = jnp.pad(Xb_grid, ((0, pad), (0, 0)))
        taper_p = jnp.pad(taper, ((0, pad), (0, 0)))
        Xb_c = Xb_p.reshape(n_chunks, chunk, K)
        taper_c = taper_p.reshape(n_chunks, chunk, taper.shape[1])

        def _block(args):
            xb_blk, taper_blk = args
            return jax.vmap(_lane)(xb_blk, taper_blk)          # (chunk, K)

        Xa_c = jax.lax.map(_block, (Xb_c, taper_c))            # (n_chunks, chunk, K)
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
        taper = self._build_taper(obs_loc_indices, dtype)

        # 2. Fused per-gridpoint local ETKF -> analysis grid columns.
        Xa_grid = self._local_columns(
            Xb_grid, Yb, Y, rinv_diag, taper, rho)             # (grid_dim, K)

        # 3. SHT project back to spectral; split mean + perturbations so the
        #    relaxation/inflation acts on the SPECTRAL perturbations (RTPS is a
        #    nonlinear per-coord rescale that does NOT commute with the SHT).
        Xa_spec = jax.vmap(self.from_grid, in_axes=1, out_axes=1)(Xa_grid)
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
