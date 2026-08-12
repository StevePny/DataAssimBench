"""Tests for the LETKF / LETKF4D Data Assimilation Cyclers.

(dabench.dacycler._letkf / dabench.dacycler._letkf4d)
"""

import pytest
import numpy as np
import jax
import jax.numpy as jnp
import jax.random as jrand
import dabench as dab
from dabench.dacycler import ETKF, LETKF, LETKF4D
from dabench.dacycler._letkf import _gaspari_cohn, _spd_inv_sqrt_ns

jax.config.update("jax_enable_x64", True)

key = jrand.PRNGKey(42)


@pytest.fixture
def l96_nature_run():
    l96 = dab.data.Lorenz96(system_dim=5, store_as_jax=True, delta_t=0.01)
    return l96.generate(n_steps=120)


@pytest.fixture
def obs_vec_l96(l96_nature_run):
    obs_l96 = dab.observer.Observer(
        l96_nature_run,
        times=l96_nature_run['time'].data[np.arange(0, 120, 5)],
        random_location_count=3,
        error_bias=0.0,
        error_sd=1.0,
        random_seed=91,
        stationary_observers=True,
        store_as_jax=True,
    )
    return obs_l96.observe()


@pytest.fixture
def l96_fc_model():
    model_l96 = dab.data.Lorenz96(system_dim=5, store_as_jax=True, delta_t=0.01)

    class L96Model(dab.model.Model):
        def forecast(self, state_vec, n_steps):
            new_vec = self.model_obj.generate(
                x0=state_vec['x'].data, n_steps=n_steps)
            return new_vec.isel(time=-1).assign_attrs(delta_t=0.01), new_vec

    return L96Model(model_obj=model_l96)


# ── (c) Gaspari-Cohn taper shape ──────────────────────────────────────────
def test_gaspari_cohn_shape():
    c = 2.0
    assert float(_gaspari_cohn(jnp.array(0.0), c)) == pytest.approx(1.0)
    # Exactly zero beyond the 2c support.
    assert float(_gaspari_cohn(jnp.array(2 * c), c)) == pytest.approx(0.0,
                                                                      abs=1e-12)
    assert float(_gaspari_cohn(jnp.array(3 * c), c)) == 0.0
    # Monotone-decreasing, bounded to [0, 1], and continuous at the r=1 seam.
    d = jnp.linspace(0.0, 2 * c, 200)
    w = _gaspari_cohn(d, c)
    assert bool(jnp.all(w >= 0.0)) and bool(jnp.all(w <= 1.0))
    assert bool(jnp.all(jnp.diff(w) <= 1e-9))
    left = float(_gaspari_cohn(jnp.array(c - 1e-4), c))
    right = float(_gaspari_cohn(jnp.array(c + 1e-4), c))
    assert abs(left - right) < 1e-3            # C1-smooth seam at r=1


# ── (a) LETKF(radius -> inf, identity transform) == global ETKF (fp64) ─────
def _direct_analysis_inputs():
    rng = np.random.default_rng(0)
    system_dim, ens, n_obs = 6, 8, 4
    Xb = jnp.asarray(rng.standard_normal((system_dim, ens)))
    obs_idx = np.array([0, 2, 3, 5])
    H = jnp.asarray(np.eye(system_dim)[obs_idx])          # (n_obs, sys)
    Y = jnp.asarray(rng.standard_normal((1, n_obs)))
    R = jnp.identity(n_obs) * (1.5 ** 2)
    return Xb, H, Y, R, obs_idx, system_dim, ens


def test_letkf_matches_etkf_radius_inf_fp64():
    Xb, H, Y, R, obs_idx, sysd, ens = _direct_analysis_inputs()
    etkf = ETKF(system_dim=sysd, delta_t=0.01, ensemble_dim=ens, model_obj=None)
    letkf = LETKF(system_dim=sysd, delta_t=0.01, ensemble_dim=ens,
                  model_obj=None, localize_radius=1e12)   # no taper (all ~1)
    Xa_e = etkf._compute_analysis(Xb=Xb, Y=Y, H=H, h=None, R=R)
    Xa_l = letkf._compute_analysis(Xb=Xb, Y=Y, H=H, h=None, R=R)
    assert Xa_l.dtype == jnp.float64
    assert np.allclose(np.asarray(Xa_e), np.asarray(Xa_l), atol=1e-9)


# ── (d) dtype purity: f32 in (under x64) -> f32 analysis ───────────────────
def test_letkf_dtype_purity_fp32():
    Xb, H, Y, R, obs_idx, sysd, ens = _direct_analysis_inputs()
    letkf = LETKF(system_dim=sysd, delta_t=0.01, ensemble_dim=ens,
                  model_obj=None, localize_radius=1e12)
    Xa = letkf._compute_analysis(
        Xb=Xb.astype(jnp.float32), Y=Y.astype(jnp.float32),
        H=H.astype(jnp.float32), h=None, R=R.astype(jnp.float32))
    assert Xa.dtype == jnp.float32            # no bare-constructor promotion


# ── (e) no-obs gridpoint (all-zero taper row) == background ────────────────
def test_letkf_no_obs_gridpoint_is_background():
    Xb, H, Y, R, obs_idx, sysd, ens = _direct_analysis_inputs()
    # Tiny radius so distant grid points see an all-zero taper row.
    letkf = LETKF(system_dim=sysd, delta_t=0.01, ensemble_dim=ens,
                  model_obj=None, localize_radius=0.4)
    Xa = letkf._compute_analysis(Xb=Xb, Y=Y, H=H, h=None, R=R)
    # Grid point 4 is >= 2c (= 0.8) from every obs index {0,2,3,5} on the ring
    # of length 6 (dists to 4: 2,2,1,1 -> min 1 > 0.8) -> untouched background.
    assert np.allclose(np.asarray(Xa[4]), np.asarray(Xb[4]), atol=1e-10)


# ── L96 cycling helpers ────────────────────────────────────────────────────
def _l96_init(l96_nature_run, ens=4, cur_tstep=10):
    init_noise = jrand.normal(key, shape=(ens, 5))
    init_state = l96_nature_run.isel(time=cur_tstep)
    return init_state.assign(
        x=(['ensemble', 'index'], init_state['x'].data + init_noise))


def _run_cycle(cycler, init_state, obs_vec, obs_error_sd=1.0, n_cycles=10):
    return cycler.cycle(
        input_state=init_state, start_time=init_state['time'].data,
        obs_vector=obs_vec, obs_error_sd=obs_error_sd, analysis_window=0.1,
        n_cycles=n_cycles, return_forecast=True)


def _analysis_rmse(out, l96_nature_run, cur_tstep=10, steps_per_cycle=10):
    """Ensemble-mean analysis RMSE vs nature at each cycle's incoming IC.

    ``cycle_timestep=0`` of each cycle is that cycle's incoming IC -- the
    previous cycle's window-end analysis (default filter placement).
    """
    ana = np.asarray(out.isel(cycle_timestep=0).mean('ensemble')['x'].data)
    nat = np.asarray(l96_nature_run['x'].data)
    idx = [cur_tstep + c * steps_per_cycle for c in range(ana.shape[0])]
    return float(np.sqrt(np.mean((ana - nat[idx]) ** 2)))


# ── (b) 3D LETKF L96 cycle: shapes, finite, distinct members, DA < no-op ────
def test_letkf_l96_cycle(l96_nature_run, obs_vec_l96, l96_fc_model):
    init_state = _l96_init(l96_nature_run)
    letkf = LETKF(system_dim=5, delta_t=0.01, ensemble_dim=4,
                  model_obj=l96_fc_model, localize_radius=1.5)
    out = _run_cycle(letkf, init_state, obs_vec_l96)

    assert out['x'].shape == (10, 4, 10, 5)
    stacked = out.stack(time=['cycle', 'cycle_timestep']).transpose('time', ...)
    assert stacked['x'].shape == (100, 4, 5)
    assert bool(np.all(np.isfinite(np.asarray(out['x'].data))))
    # Transform keeps ensemble members distinct.
    assert not jnp.allclose(out['x'].values[-1, 1, 0, :],
                            out['x'].values[-1, 0, 0, :])
    # Domain-localized DA pulls the analysis toward truth vs a near no-op R.
    rmse_da = _analysis_rmse(out, l96_nature_run)
    letkf_noop = LETKF(system_dim=5, delta_t=0.01, ensemble_dim=4,
                       model_obj=l96_fc_model, localize_radius=1.5)
    rmse_noop = _analysis_rmse(
        _run_cycle(letkf_noop, init_state, obs_vec_l96, obs_error_sd=1.0e6),
        l96_nature_run)
    assert rmse_da < rmse_noop


# ── (b) 4D LETKF L96 cycle: shapes, finite, distinct members, DA < no-op ────
def test_letkf4d_l96_cycle(l96_nature_run, obs_vec_l96, l96_fc_model):
    init_state = _l96_init(l96_nature_run)
    letkf = LETKF4D(system_dim=5, delta_t=0.01, ensemble_dim=4,
                    model_obj=l96_fc_model, localize_radius=1.5)
    out = _run_cycle(letkf, init_state, obs_vec_l96)

    assert out['x'].shape == (10, 4, 10, 5)
    assert bool(np.all(np.isfinite(np.asarray(out['x'].data))))
    assert not jnp.allclose(out['x'].values[-1, 1, 0, :],
                            out['x'].values[-1, 0, 0, :])
    rmse_da = _analysis_rmse(out, l96_nature_run)
    letkf_noop = LETKF4D(system_dim=5, delta_t=0.01, ensemble_dim=4,
                         model_obj=l96_fc_model, localize_radius=1.5)
    rmse_noop = _analysis_rmse(
        _run_cycle(letkf_noop, init_state, obs_vec_l96, obs_error_sd=1.0e6),
        l96_nature_run)
    assert rmse_da < rmse_noop


# ── (f) window-stacked taper alignment with an injected obs_latlon ──────────
def test_letkf_windowstacked_taper_matches_obs_axis():
    """LETKF4D window-stacks the obs axis (``n_times * obs_dim``) while the
    driver injects a single-slice ``obs_latlon`` (``obs_dim`` rows).  The taper
    obs axis MUST track the (window-stacked) obs vector; the injected slice is
    tiled ``n_times`` times.  Regression for the ``(grid_dim, obs_dim)`` vs
    ``(n_times*obs_dim,)`` broadcast crash seen at T42."""
    rng = np.random.default_rng(3)
    grid_dim, obs_dim, n_times = 6, 4, 3
    grid_latlon = jnp.asarray(
        np.stack([rng.uniform(-80, 80, grid_dim),
                  rng.uniform(0, 360, grid_dim)], axis=1))
    obs_idx = np.array([0, 2, 3, 5])
    obs_latlon = grid_latlon[obs_idx]                     # (obs_dim, 2) slice
    letkf = LETKF(system_dim=grid_dim, delta_t=0.01, ensemble_dim=4,
                  model_obj=None, grid_latlon=grid_latlon,
                  obs_latlon=obs_latlon, localize_radius=800.0)

    # Single-slice obs axis -> (grid_dim, obs_dim), tile factor 1.
    taper_1 = letkf._build_taper(jnp.asarray(obs_idx), jnp.float64)
    assert taper_1.shape == (grid_dim, obs_dim)

    # Window-stacked obs axis (n_times slices) -> (grid_dim, n_times*obs_dim).
    letkf._taper_cache = None                             # bypass the cache
    stacked_idx = jnp.asarray(np.tile(obs_idx, n_times))  # (n_times*obs_dim,)
    taper_n = letkf._build_taper(stacked_idx, jnp.float64)
    assert taper_n.shape == (grid_dim, n_times * obs_dim)
    # Each tiled block equals the single-slice taper (same stationary geometry).
    for t in range(n_times):
        blk = taper_n[:, t * obs_dim:(t + 1) * obs_dim]
        assert np.allclose(np.asarray(blk), np.asarray(taper_1))

    # Full localized analysis on the window-stacked innovation must broadcast
    # cleanly (this is the exact path that crashed) and stay finite.
    ens = 4
    Xb = jnp.asarray(rng.standard_normal((grid_dim, ens)))
    Yb = jnp.asarray(rng.standard_normal((n_times * obs_dim, ens)))
    Y = jnp.asarray(rng.standard_normal(n_times * obs_dim))
    rinv = jnp.asarray(rng.uniform(0.5, 2.0, n_times * obs_dim))
    letkf._taper_cache = None
    Xa = letkf._localized_analysis(Xb, Yb, Y, rinv, stacked_idx, rho=1.0)
    assert Xa.shape == (grid_dim, ens)
    assert bool(np.all(np.isfinite(np.asarray(Xa))))


def test_letkf_grid_chunk_matches_whole_grid():
    """Grid-chunking the per-gridpoint solve must match the whole-grid ``vmap``
    to round-off (same lane, same order; ``lax.map`` vs ``vmap`` only reorders
    XLA fusion) -- it bounds the peak ``(chunk, K, n_obs)`` intermediate so
    T42-4D fits in GPU memory.  Exercises a chunk that does NOT divide the grid
    (padding path)."""
    rng = np.random.default_rng(7)
    grid_dim, n_obs, ens = 13, 20, 4          # 13 % 5 != 0 -> padding path
    grid_latlon = jnp.asarray(
        np.stack([rng.uniform(-80, 80, grid_dim),
                  rng.uniform(0, 360, grid_dim)], axis=1))
    obs_latlon = jnp.asarray(
        np.stack([rng.uniform(-80, 80, n_obs),
                  rng.uniform(0, 360, n_obs)], axis=1))
    Xb = jnp.asarray(rng.standard_normal((grid_dim, ens)))
    Yb = jnp.asarray(rng.standard_normal((n_obs, ens)))
    Y = jnp.asarray(rng.standard_normal(n_obs))
    rinv = jnp.asarray(rng.uniform(0.5, 2.0, n_obs))

    def _run(chunk):
        c = LETKF(system_dim=grid_dim, delta_t=0.01, ensemble_dim=ens,
                  model_obj=None, grid_latlon=grid_latlon,
                  obs_latlon=obs_latlon, localize_radius=5000.0,
                  grid_chunk=chunk)
        taper = c._build_taper(jnp.arange(n_obs), jnp.float64)
        return np.asarray(c._local_columns(Xb, Yb, Y, rinv, taper, rho=1.0))

    whole = _run(None)
    for chunk in (1, 5, 8, 13, 100):
        assert np.allclose(_run(chunk), whole, rtol=0, atol=1e-12), (
            f"chunk={chunk} differs beyond round-off")


def test_letkf_eigh_impl_invalid_raises():
    """``eigh_impl`` must be None/'qr'/'jacobi'; anything else fails loudly at
    construction (a typo would otherwise silently fall through to the default)."""
    with pytest.raises(ValueError, match="eigh_impl"):
        LETKF(system_dim=5, delta_t=0.01, ensemble_dim=4, model_obj=None,
              eigh_impl="lapack")


@pytest.mark.skipif(
    not hasattr(jax.lax.linalg, "EighImplementation"),
    reason="jax.lax.linalg.eigh(implementation=...) requires jax>=0.8.1")
def test_letkf_eigh_impl_qr_matches_default():
    """Forcing the QR backend via ``jax.lax.linalg.eigh`` (eigh_impl='qr') must
    match the default ``jnp.linalg.eigh`` path (eigh_impl=None) to round-off:
    same SPD eigendecomposition, only the API/return-order differs.  JACOBI is
    GPU/TPU-only so it is exercised on-device, not in CPU CI."""
    rng = np.random.default_rng(11)
    grid_dim, n_obs, ens = 9, 16, 6
    grid_latlon = jnp.asarray(
        np.stack([rng.uniform(-80, 80, grid_dim),
                  rng.uniform(0, 360, grid_dim)], axis=1))
    obs_latlon = jnp.asarray(
        np.stack([rng.uniform(-80, 80, n_obs),
                  rng.uniform(0, 360, n_obs)], axis=1))
    Xb = jnp.asarray(rng.standard_normal((grid_dim, ens)))
    Yb = jnp.asarray(rng.standard_normal((n_obs, ens)))
    Y = jnp.asarray(rng.standard_normal(n_obs))
    rinv = jnp.asarray(rng.uniform(0.5, 2.0, n_obs))

    def _run(impl):
        c = LETKF(system_dim=grid_dim, delta_t=0.01, ensemble_dim=ens,
                  model_obj=None, grid_latlon=grid_latlon,
                  obs_latlon=obs_latlon, localize_radius=5000.0,
                  grid_chunk=None, eigh_impl=impl)
        taper = c._build_taper(jnp.arange(n_obs), jnp.float64)
        return np.asarray(c._local_columns(Xb, Yb, Y, rinv, taper, rho=1.0))

    assert np.allclose(_run("qr"), _run(None), rtol=0, atol=1e-10)


def test_spd_inv_sqrt_ns_matches_eigh():
    """The eigh-free Newton-Schulz A^{-1/2} must match the eigh-based inverse
    square root to round-off for a batch of well-conditioned SPD matrices, and
    satisfy Z A Z == I.  This is the GPU-friendly (matmul-only) solver that
    replaces the host-bound batched eigh at K=64."""
    rng = np.random.default_rng(23)
    B, K = 40, 64
    # SPD batch A = M M^T + c I  (c keeps the spectrum well away from 0).
    M = rng.standard_normal((B, K, K))
    A = np.einsum("bij,bkj->bik", M, M) + 2.0 * np.eye(K)[None]
    A = jnp.asarray(A, dtype=jnp.float64)

    Z = _spd_inv_sqrt_ns(A, n_iter=20)
    # eigh reference: A^{-1/2} = V diag(1/sqrt(w)) V^T.
    w, V = jnp.linalg.eigh(A)
    ref = jnp.einsum("bij,bj,bkj->bik", V, 1.0 / jnp.sqrt(w), V)
    assert np.allclose(np.asarray(Z), np.asarray(ref), rtol=0, atol=1e-9)
    # Z A Z == I (defining property of the inverse square root).
    ZAZ = jnp.einsum("bij,bjk,bkl->bil", Z, A, Z)
    assert np.allclose(np.asarray(ZAZ),
                       np.asarray(jnp.eye(K))[None], rtol=0, atol=1e-9)


def test_letkf_newton_schulz_matches_default():
    """LETKF with eigh_impl='newton_schulz' (eigh-free, matmul-only) must match
    the default eigh path (eigh_impl=None) to round-off across the whole
    per-gridpoint local solve -- the equivalence that lets the GPU run use NS
    for a genuinely batched (GEMM) solve while CPU/L96 keep eigh."""
    rng = np.random.default_rng(29)
    grid_dim, n_obs, ens = 9, 16, 6
    grid_latlon = jnp.asarray(
        np.stack([rng.uniform(-80, 80, grid_dim),
                  rng.uniform(0, 360, grid_dim)], axis=1))
    obs_latlon = jnp.asarray(
        np.stack([rng.uniform(-80, 80, n_obs),
                  rng.uniform(0, 360, n_obs)], axis=1))
    Xb = jnp.asarray(rng.standard_normal((grid_dim, ens)))
    Yb = jnp.asarray(rng.standard_normal((n_obs, ens)))
    Y = jnp.asarray(rng.standard_normal(n_obs))
    rinv = jnp.asarray(rng.uniform(0.5, 2.0, n_obs))

    def _run(impl):
        c = LETKF(system_dim=grid_dim, delta_t=0.01, ensemble_dim=ens,
                  model_obj=None, grid_latlon=grid_latlon,
                  obs_latlon=obs_latlon, localize_radius=5000.0,
                  grid_chunk=None, eigh_impl=impl)
        taper = c._build_taper(jnp.arange(n_obs), jnp.float64)
        return np.asarray(c._local_columns(Xb, Yb, Y, rinv, taper, rho=1.0))

    assert np.allclose(_run("newton_schulz"), _run(None), rtol=0, atol=1e-8)
    # 'ns' is an accepted alias for 'newton_schulz'.
    assert np.allclose(_run("ns"), _run(None), rtol=0, atol=1e-8)


def test_letkf_obs_axis_non_multiple_raises():
    """A window-stacked obs axis that is not an integer multiple of the injected
    ``obs_latlon`` slice is an alignment error and must fail loudly."""
    grid_dim, obs_dim = 6, 4
    grid_latlon = jnp.asarray(
        np.stack([np.linspace(-80, 80, grid_dim),
                  np.linspace(0, 300, grid_dim)], axis=1))
    obs_idx = np.array([0, 2, 3, 5])
    letkf = LETKF(system_dim=grid_dim, delta_t=0.01, ensemble_dim=4,
                  model_obj=None, grid_latlon=grid_latlon,
                  obs_latlon=grid_latlon[obs_idx], localize_radius=800.0)
    with pytest.raises(ValueError, match="integer multiple"):
        letkf._build_taper(jnp.arange(obs_dim + 1), jnp.float64)


# ── taper cache must not leak a tracer: the cycler is reusable ──────────────
def test_letkf_cycler_reusable_no_tracer_leak(l96_nature_run, obs_vec_l96,
                                              l96_fc_model):
    """The Gaspari-Cohn taper is built under ``jax.lax.scan``; it must be
    recomputed (not cached as a tracer) so a second ``.cycle()`` on the same
    cycler does not raise ``UnexpectedTracerError``."""
    init_state = _l96_init(l96_nature_run)
    for cls in (LETKF, LETKF4D):
        cycler = cls(system_dim=5, delta_t=0.01, ensemble_dim=4,
                     model_obj=l96_fc_model, localize_radius=1.5)
        out1 = _run_cycle(cycler, init_state, obs_vec_l96)
        out2 = _run_cycle(cycler, init_state, obs_vec_l96)   # was: tracer leak
        assert np.allclose(np.asarray(out1['x'].data),
                           np.asarray(out2['x'].data))
