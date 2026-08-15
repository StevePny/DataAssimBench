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
from dabench.dacycler._letkf import (
    _gaspari_cohn, _great_circle_km, _spd_inv_sqrt_ns, build_patch_geometry,
    build_patch_geometry_series, PatchGeometryProducer)

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
def obs_vec_l96_observable(l96_nature_run):
    """Fully-observable obs network for the cycle-skill tests.

    The default ``obs_vec_l96`` (3-of-5 locations, ens=4) sits in a marginal
    observability regime (``k-1`` and the observed DOF barely span the L96(5)
    unstable subspace), so DA-vs-no-op skill there tests the boundary, not the
    filter.  This fixture observes ALL 5 variables at a modest error so the
    cycle tests validate the DA itself (paired with a larger ensemble).
    """
    obs_l96 = dab.observer.Observer(
        l96_nature_run,
        times=l96_nature_run['time'].data[np.arange(0, 120, 5)],
        random_location_count=5,
        error_bias=0.0,
        error_sd=0.5,
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


# ── (b) 3D-FGAT LETKF L96 cycle: shapes, finite, distinct members, DA < no-op ─
def test_letkf_l96_cycle(l96_nature_run, obs_vec_l96_observable, l96_fc_model):
    # The analysis window (0.1) spans MULTIPLE obs times (obs every 0.05), so a
    # plain 3D LETKF collapses all in-window obs onto one static state and
    # cannot fit them (structurally broken for multi-time windows).  3D-FGAT
    # forms innovations at each obs's true time while building the increment at
    # the single analysis time -- the correct 3D treatment.  Paired with an
    # observable network (5/5 obs) + a sufficient ensemble so this validates the
    # DA, not the observability boundary.
    ens = 10
    init_state = _l96_init(l96_nature_run, ens=ens)
    letkf = LETKF(system_dim=5, delta_t=0.01, ensemble_dim=ens,
                  model_obj=l96_fc_model, localize_radius=1.5, fgat=True)
    out = _run_cycle(letkf, init_state, obs_vec_l96_observable)

    assert out['x'].shape == (10, ens, 10, 5)
    stacked = out.stack(time=['cycle', 'cycle_timestep']).transpose('time', ...)
    assert stacked['x'].shape == (100, ens, 5)
    assert bool(np.all(np.isfinite(np.asarray(out['x'].data))))
    # Transform keeps ensemble members distinct.
    assert not jnp.allclose(out['x'].values[-1, 1, 0, :],
                            out['x'].values[-1, 0, 0, :])
    # Domain-localized FGAT DA pulls the analysis toward truth vs a near no-op R.
    rmse_da = _analysis_rmse(out, l96_nature_run)
    letkf_noop = LETKF(system_dim=5, delta_t=0.01, ensemble_dim=ens,
                       model_obj=l96_fc_model, localize_radius=1.5, fgat=True)
    rmse_noop = _analysis_rmse(
        _run_cycle(letkf_noop, init_state, obs_vec_l96_observable,
                   obs_error_sd=1.0e6),
        l96_nature_run)
    assert rmse_da < rmse_noop


# ── (b) 4D LETKF L96 cycle: shapes, finite, distinct members, DA < no-op ────
def test_letkf4d_l96_cycle(l96_nature_run, obs_vec_l96_observable,
                           l96_fc_model):
    ens = 10
    init_state = _l96_init(l96_nature_run, ens=ens)
    letkf = LETKF4D(system_dim=5, delta_t=0.01, ensemble_dim=ens,
                    model_obj=l96_fc_model, localize_radius=1.5)
    out = _run_cycle(letkf, init_state, obs_vec_l96_observable)

    assert out['x'].shape == (10, ens, 10, 5)
    assert bool(np.all(np.isfinite(np.asarray(out['x'].data))))
    assert not jnp.allclose(out['x'].values[-1, 1, 0, :],
                            out['x'].values[-1, 0, 0, :])
    rmse_da = _analysis_rmse(out, l96_nature_run)
    letkf_noop = LETKF4D(system_dim=5, delta_t=0.01, ensemble_dim=ens,
                         model_obj=l96_fc_model, localize_radius=1.5)
    rmse_noop = _analysis_rmse(
        _run_cycle(letkf_noop, init_state, obs_vec_l96_observable,
                   obs_error_sd=1.0e6),
        l96_nature_run)
    assert rmse_da < rmse_noop


# ── capture_first_transforms: real cycle-0 A stack == analysis transform ────
def test_letkf4d_capture_first_transforms(l96_nature_run,
                                          obs_vec_l96_observable,
                                          l96_fc_model):
    """``LETKF4D.capture_first_transforms`` must return the EXACT per-gridpoint
    SPD transforms ``A`` the first analysis solves: a concrete
    ``(grid_dim, K, K)`` SPD stack that, when solved + recombined the same way
    ``_local_columns`` does on the same cycle-0 inputs, reproduces the cycler's
    own analysis columns to round-off.  This is the offline solver-diagnostic
    capture path (replay real transforms through other backends/precisions)."""
    ens = 10
    init_state = _l96_init(l96_nature_run, ens=ens)
    letkf = LETKF4D(system_dim=5, delta_t=0.01, ensemble_dim=ens,
                    model_obj=l96_fc_model, localize_radius=1.5)
    kw = dict(input_state=init_state, start_time=init_state['time'].data,
              obs_vector=obs_vec_l96_observable, obs_error_sd=0.5,
              analysis_window=0.1, n_cycles=10)

    A = letkf.capture_first_transforms(**kw)
    assert A.ndim == 3 and A.shape[1] == A.shape[2] == ens
    assert np.isfinite(A).all()
    w = np.linalg.eigvalsh(A)                          # SPD: all eigs > 0
    assert float(w.min()) > 0.0

    # Rebuild the SAME cycle-0 inputs the capture used, so _local_columns is
    # called on identical (Xb_grid, Yb, Y, rinv, taper), then reconstruct the
    # analysis from the captured A and compare to the cycler's own output.
    inp, allpad = letkf._prepare_cycle(
        init_state, init_state['time'].data, obs_vec_l96_observable, 0.5, 10,
        0.1, None)
    cur_time = jnp.asarray(inp['_cur_time'].data)
    cur_state = inp.drop_vars(['_cur_time'])
    fidx = jnp.asarray(allpad[0]) - 1
    otm = jnp.asarray(allpad[0]) > 0
    cov = jnp.array(letkf._obs_vector[letkf._observed_vars]
                    .to_stacked_array('system', ['time']).data).at[fidx].get()
    cot = jnp.array(letkf._obs_vector.time.data).at[fidx].get()
    coli = jnp.array(letkf._obs_vector.system_index.data).at[:, fidx].get(
        ).reshape(fidx.shape[0], -1)
    colm = jnp.array(letkf._obs_loc_masks).at[:, fidx].get().astype(bool
        ).reshape(fidx.shape[0], -1)
    owi = jnp.array([jnp.argmin(jnp.abs(t - (cur_time + letkf._model_timesteps)))
                     for t in cot])
    _, fc = letkf._step_forecast(cur_state, n_steps=letkf.steps_per_window)
    Xtraj, Yb, Y, rinv = letkf._build_yb(fc, cov, coli, otm, colm, owi)
    obs_loc_flat = jnp.asarray(coli).reshape(-1)
    tau = letkf._resolve_analysis_index()
    Xb_tau = Xtraj[:, tau, :].T
    dtype = Xb_tau.dtype
    Xb_grid = jax.vmap(letkf.to_grid, in_axes=1, out_axes=1)(Xb_tau)
    taper = letkf._build_taper(obs_loc_flat, dtype)
    Xa_ref = np.asarray(letkf._local_columns(
        Xb_grid, Yb, Y, rinv, taper, rho=letkf.multiplicative_inflation))

    from dabench.dacycler._utils import _solve_pa_wa
    K = ens
    Iden = jnp.identity(K, dtype=dtype)
    U = jnp.ones((K, K), dtype=dtype) / K
    Yb_pert = Yb @ (Iden - U)
    innov = (Y - jnp.mean(Yb, axis=1)).astype(dtype)
    A_j = jnp.asarray(A)

    def _recon(g):
        Pa, Wa, _ = _solve_pa_wa(A_j[g], letkf.eigh_impl, letkf.ns_iters)
        YtRinv = Yb_pert.T * (
            rinv.astype(dtype) * taper[g].astype(dtype))[None, :]
        wa = Pa @ (YtRinv @ innov)
        xb_col = Xb_grid[g]
        xb_bar = jnp.mean(xb_col)
        xb_pert = xb_col - xb_bar
        return xb_pert @ Wa + xb_bar + jnp.dot(xb_pert, wa)

    Xa_cap = np.asarray(jax.vmap(_recon)(jnp.arange(A.shape[0])))
    assert np.allclose(Xa_cap, Xa_ref, rtol=0, atol=1e-9)


# ── obs-space metrics: LETKF (inherits ETKF._cycle_obsop) + LETKF4D ─────────
@pytest.mark.parametrize("cls", [LETKF, LETKF4D])
def test_letkf_obs_metrics(l96_nature_run, obs_vec_l96, l96_fc_model, cls):
    init_state = _l96_init(l96_nature_run)
    cycler = cls(system_dim=5, delta_t=0.01, ensemble_dim=4,
                 model_obj=l96_fc_model, localize_radius=1.5)
    kw = dict(input_state=init_state, start_time=init_state['time'].data,
              obs_vector=obs_vec_l96, obs_error_sd=1.0, analysis_window=0.1,
              n_cycles=10, return_forecast=True)

    ana_default = cls(system_dim=5, delta_t=0.01, ensemble_dim=4,
                      model_obj=l96_fc_model, localize_radius=1.5).cycle(**kw)
    ana, metrics = cycler.cycle(return_metrics=True, **kw)
    # Byte-identical analysis regardless of return_metrics.
    assert np.array_equal(np.asarray(ana_default['x'].data),
                          np.asarray(ana['x'].data))
    for var in ("o_minus_f_rms", "o_minus_a_rms", "bias_f", "bias_a",
                "obs_space_spread_background", "sigma_obs_max",
                "n_active_obs"):
        assert metrics[var].shape == (10,)
        assert np.asarray(metrics[var].data).dtype == np.float64
    n_active = np.asarray(metrics["n_active_obs"].data)
    act = n_active > 0
    assert bool(np.any(act))
    spread = np.asarray(metrics["obs_space_spread_background"].data)
    assert bool(np.all(np.isfinite(spread[act])))
    assert bool(np.all(spread[act] > 0))

    _, metrics_dbg = cycler.cycle(
        return_metrics=True, metrics_mode="debug", **kw)
    for var in ("o_minus_f", "o_minus_a", "obs_active"):
        assert metrics_dbg[var].dims == ("cycle", "obs")


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


# ── Regime-A local-patch gather (§14.3) ────────────────────────────────────
def test_build_patch_geometry_exact_and_truncated():
    """``build_patch_geometry`` must recover, per grid point, the SAME nearest
    obs + Gaspari-Cohn weights the dense taper carries.  At ``P=max`` neighbour
    count nothing is truncated; a smaller ``P`` keeps the NEAREST obs and logs
    the discarded weight mass."""
    rng = np.random.default_rng(101)
    grid_dim, n_obs = 12, 25
    grid_latlon = np.stack([rng.uniform(-70, 70, grid_dim),
                            rng.uniform(0, 360, grid_dim)], axis=1)
    obs_latlon = np.stack([rng.uniform(-70, 70, n_obs),
                           rng.uniform(0, 360, n_obs)], axis=1)
    R = 3000.0                                       # km; wide enough to overlap

    patch_idx, patch_gc, diag = build_patch_geometry(
        grid_latlon, obs_latlon, localize_radius=R, localize_units="km")
    assert patch_idx.shape == patch_gc.shape
    assert patch_idx.dtype == np.int32
    # Exact P => no truncation, no discarded mass.
    assert diag["pct_truncated"] == 0.0
    assert diag["weight_mass_discarded"] == 0.0
    # Every in-patch nonzero weight matches the dense GC of that (grid, obs) pair.
    dense = np.asarray(_gaspari_cohn(
        np.asarray(_great_circle_km(
            grid_latlon[:, 0][:, None], grid_latlon[:, 1][:, None],
            obs_latlon[:, 0][None, :], obs_latlon[:, 1][None, :])), R))
    for g in range(grid_dim):
        for p in range(patch_idx.shape[1]):
            w = patch_gc[g, p]
            if w > 0:
                assert np.isclose(w, dense[g, patch_idx[g, p]], atol=1e-12)

    # Truncated P: keeps the P nearest, reports discarded mass >= 0.
    idx_t, gc_t, diag_t = build_patch_geometry(
        grid_latlon, obs_latlon, localize_radius=R, localize_units="km",
        patch_size=2)
    assert idx_t.shape[1] == 2
    assert diag_t["weight_mass_discarded"] >= 0.0


def test_letkf_patch_gather_matches_dense():
    """The sparse local-patch gather (Regime A, §14.3) must reproduce the dense
    Gaspari-Cohn taper analysis to round-off when ``P`` covers every neighbour:
    identical GC weights + validity, just gathered instead of broadcast.  This
    is the equivalence that lets the patch API replace the ``(grid_dim, n_obs)``
    taper without changing the analysis."""
    rng = np.random.default_rng(202)
    grid_dim, n_obs, ens = 15, 30, 5
    grid_latlon = jnp.asarray(
        np.stack([rng.uniform(-70, 70, grid_dim),
                  rng.uniform(0, 360, grid_dim)], axis=1))
    obs_latlon = jnp.asarray(
        np.stack([rng.uniform(-70, 70, n_obs),
                  rng.uniform(0, 360, n_obs)], axis=1))
    Xb = jnp.asarray(rng.standard_normal((grid_dim, ens)))
    Yb = jnp.asarray(rng.standard_normal((n_obs, ens)))
    Y = jnp.asarray(rng.standard_normal(n_obs))
    rinv = jnp.asarray(rng.uniform(0.5, 2.0, n_obs))
    R = 2500.0

    dense = LETKF(system_dim=grid_dim, delta_t=0.01, ensemble_dim=ens,
                  model_obj=None, grid_latlon=grid_latlon,
                  obs_latlon=obs_latlon, localize_radius=R, grid_chunk=None)
    taper = dense._build_taper(jnp.arange(n_obs), jnp.float64)
    Xa_dense = np.asarray(dense._local_columns(Xb, Yb, Y, rinv, taper, rho=1.0))

    patch_idx, patch_gc, _ = build_patch_geometry(
        np.asarray(grid_latlon), np.asarray(obs_latlon), localize_radius=R)
    for chunk in (None, 4, 15):
        patch = LETKF(system_dim=grid_dim, delta_t=0.01, ensemble_dim=ens,
                      model_obj=None, grid_latlon=grid_latlon,
                      obs_latlon=obs_latlon, localize_radius=R,
                      grid_chunk=chunk, patch_idx=patch_idx, patch_gc=patch_gc)
        Xa_patch = np.asarray(
            patch._local_columns(Xb, Yb, Y, rinv, None, rho=1.0))
        assert np.allclose(Xa_patch, Xa_dense, rtol=0, atol=1e-10), (
            f"patch chunk={chunk} differs from dense taper beyond round-off")

    # capture_A_matrices patch path must match the dense A-stack too.
    Xb_spec = jnp.asarray(rng.standard_normal((grid_dim, ens)))
    A_dense = np.asarray(dense.capture_A_matrices(
        Xb_spec, Yb, Y, rinv, jnp.arange(n_obs), rho=1.0))
    A_patch = np.asarray(patch.capture_A_matrices(
        Xb_spec, Yb, Y, rinv, jnp.arange(n_obs), rho=1.0))
    assert np.allclose(A_patch, A_dense, rtol=0, atol=1e-10)


def test_letkf_patch_no_obs_gridpoint_is_background():
    """A grid point whose patch is entirely beyond ``2c`` (all-zero patch_w)
    must fall through to ``A=(K-1)/rho I`` -> analysis = background, exactly like
    the dense no-obs case."""
    rng = np.random.default_rng(303)
    grid_dim, n_obs, ens = 8, 12, 4
    grid_latlon = np.stack([rng.uniform(-70, 70, grid_dim),
                            rng.uniform(0, 360, grid_dim)], axis=1)
    # Put obs far from grid point 0 by placing it at a distinct pole-ish spot.
    grid_latlon[0] = [-89.0, 0.0]
    obs_latlon = np.stack([rng.uniform(20, 70, n_obs),
                           rng.uniform(0, 360, n_obs)], axis=1)
    R = 500.0                                        # tight: gp0 sees no obs
    patch_idx, patch_gc, _ = build_patch_geometry(
        grid_latlon, obs_latlon, localize_radius=R)
    assert np.all(patch_gc[0] == 0.0)                # gp0 patch fully tapered

    Xb = jnp.asarray(rng.standard_normal((grid_dim, ens)))
    Yb = jnp.asarray(rng.standard_normal((n_obs, ens)))
    Y = jnp.asarray(rng.standard_normal(n_obs))
    rinv = jnp.asarray(rng.uniform(0.5, 2.0, n_obs))
    patch = LETKF(system_dim=grid_dim, delta_t=0.01, ensemble_dim=ens,
                  model_obj=None, grid_latlon=jnp.asarray(grid_latlon),
                  obs_latlon=jnp.asarray(obs_latlon), localize_radius=R,
                  grid_chunk=None, patch_idx=patch_idx, patch_gc=patch_gc)
    Xa = np.asarray(patch._local_columns(Xb, Yb, Y, rinv, None, rho=1.0))
    assert np.allclose(Xa[0], np.asarray(Xb[0]), atol=1e-10)


def test_letkf_patch_windowstacked_matches_dense():
    """With a window-stacked obs axis (``n_times * pool``), the patch gather must
    tile the static pool geometry per block (offset ``t*pool``) and match the
    dense tiled taper to round-off -- the 4D path."""
    rng = np.random.default_rng(404)
    grid_dim, pool, n_times, ens = 10, 6, 3, 4
    grid_latlon = np.stack([rng.uniform(-70, 70, grid_dim),
                            rng.uniform(0, 360, grid_dim)], axis=1)
    obs_latlon = np.stack([rng.uniform(-70, 70, pool),
                           rng.uniform(0, 360, pool)], axis=1)
    n_obs = n_times * pool
    Xb = jnp.asarray(rng.standard_normal((grid_dim, ens)))
    Yb = jnp.asarray(rng.standard_normal((n_obs, ens)))
    Y = jnp.asarray(rng.standard_normal(n_obs))
    rinv = jnp.asarray(rng.uniform(0.5, 2.0, n_obs))
    R = 4000.0

    dense = LETKF(system_dim=grid_dim, delta_t=0.01, ensemble_dim=ens,
                  model_obj=None, grid_latlon=jnp.asarray(grid_latlon),
                  obs_latlon=jnp.asarray(obs_latlon), localize_radius=R,
                  grid_chunk=None)
    taper = dense._build_taper(jnp.asarray(np.tile(np.arange(pool), n_times)),
                              jnp.float64)
    assert taper.shape == (grid_dim, n_obs)
    Xa_dense = np.asarray(dense._local_columns(Xb, Yb, Y, rinv, taper, rho=1.0))

    patch_idx, patch_gc, _ = build_patch_geometry(
        grid_latlon, obs_latlon, localize_radius=R)
    patch = LETKF(system_dim=grid_dim, delta_t=0.01, ensemble_dim=ens,
                  model_obj=None, grid_latlon=jnp.asarray(grid_latlon),
                  obs_latlon=jnp.asarray(obs_latlon), localize_radius=R,
                  grid_chunk=None, patch_idx=patch_idx, patch_gc=patch_gc)
    Xa_patch = np.asarray(patch._local_columns(Xb, Yb, Y, rinv, None, rho=1.0))
    assert np.allclose(Xa_patch, Xa_dense, rtol=0, atol=1e-10)


def test_letkf_patch_localized_analysis_matches_dense():
    """The FULL ``_localized_analysis`` (ISHT lift -> local solve -> SHT project
    -> relax) must match between the dense taper and the patch gather to
    round-off, so ``.cycle()`` is unaffected by the localization refactor.
    Identity transforms keep grid == spectral so the comparison is direct."""
    rng = np.random.default_rng(505)
    grid_dim, n_obs, ens = 14, 28, 6
    grid_latlon = np.stack([rng.uniform(-70, 70, grid_dim),
                            rng.uniform(0, 360, grid_dim)], axis=1)
    obs_latlon = np.stack([rng.uniform(-70, 70, n_obs),
                           rng.uniform(0, 360, n_obs)], axis=1)
    Xb = jnp.asarray(rng.standard_normal((grid_dim, ens)))
    Yb = jnp.asarray(rng.standard_normal((n_obs, ens)))
    Y = jnp.asarray(rng.standard_normal(n_obs))
    rinv = jnp.asarray(rng.uniform(0.5, 2.0, n_obs))
    obs_idx = jnp.arange(n_obs)
    R = 2600.0

    dense = LETKF(system_dim=grid_dim, delta_t=0.01, ensemble_dim=ens,
                  model_obj=None, grid_latlon=jnp.asarray(grid_latlon),
                  obs_latlon=jnp.asarray(obs_latlon), localize_radius=R,
                  grid_chunk=None)
    Xa_dense = np.asarray(dense._localized_analysis(
        Xb, Yb, Y, rinv, obs_idx, rho=1.0))

    patch_idx, patch_gc, _ = build_patch_geometry(
        grid_latlon, obs_latlon, localize_radius=R)
    patch = LETKF(system_dim=grid_dim, delta_t=0.01, ensemble_dim=ens,
                  model_obj=None, grid_latlon=jnp.asarray(grid_latlon),
                  obs_latlon=jnp.asarray(obs_latlon), localize_radius=R,
                  grid_chunk=None, patch_idx=patch_idx, patch_gc=patch_gc)
    Xa_patch = np.asarray(patch._localized_analysis(
        Xb, Yb, Y, rinv, obs_idx, rho=1.0))
    assert np.allclose(Xa_patch, Xa_dense, rtol=0, atol=1e-10)


def test_letkf_patch_idx_gc_both_or_neither():
    """``patch_idx`` and ``patch_gc`` must be supplied together (or both None)."""
    with pytest.raises(ValueError, match="together"):
        LETKF(system_dim=5, delta_t=0.01, ensemble_dim=4, model_obj=None,
              patch_idx=np.zeros((5, 2), np.int32))


# ── Regime B (per-cycle / moving-observer patch geometry, §14.4) ───────────
def _regimeB_setup(seed, G=14, npool=28, ens=6, n_cycles=4):
    """Common (grid, per-cycle obs positions, ensemble) fixture for Regime B."""
    rng = np.random.default_rng(seed)
    grid = np.stack([rng.uniform(-70, 70, G), rng.uniform(0, 360, G)], axis=1)
    series = [np.stack([rng.uniform(-70, 70, npool),
                        rng.uniform(0, 360, npool)], axis=1)
              for _ in range(n_cycles)]
    Xb = jnp.asarray(rng.standard_normal((G, ens)))
    Yb = jnp.asarray(rng.standard_normal((npool, ens)))
    Y = jnp.asarray(rng.standard_normal(npool))
    rinv = jnp.asarray(rng.uniform(0.5, 2.0, npool))
    return grid, series, Xb, Yb, Y, rinv, npool, ens


def test_letkf_patch_series_builder_row0_matches_static():
    """``build_patch_geometry_series`` row t == the static single-cycle build at
    that cycle's obs positions (same metric, same GC weights, same P)."""
    grid, series, *_ = _regimeB_setup(31)
    pidx, pgc, diag = build_patch_geometry_series(
        grid, series, localize_radius=2600.0)
    P = diag["P"]
    assert pidx.shape == (len(series), grid.shape[0], P)
    for t, o in enumerate(series):
        pi, pw, _ = build_patch_geometry(
            grid, o, localize_radius=2600.0, patch_size=P)
        assert np.array_equal(pidx[t], pi)
        assert np.allclose(pgc[t], pw, rtol=0, atol=1e-12)


def test_letkf_patch_series_producer_matches_eager():
    """The async ``PatchGeometryProducer`` yields cycle 0 immediately and a full
    stack byte-identical to the eager builder."""
    grid, series, *_ = _regimeB_setup(32)
    eager_idx, eager_gc, _ = build_patch_geometry_series(
        grid, series, localize_radius=2600.0)
    prod = build_patch_geometry_series(
        grid, series, localize_radius=2600.0, producer=True)
    assert isinstance(prod, PatchGeometryProducer)
    i0, w0, _ = prod.build0()
    assert np.array_equal(i0, eager_idx[0]) and np.allclose(w0, eager_gc[0])
    full_idx, full_gc, _ = prod.stack()
    assert np.array_equal(full_idx, eager_idx)
    assert np.allclose(full_gc, eager_gc, rtol=0, atol=1e-12)


def test_letkf_patch_series_static_pool_matches_regimeA():
    """When every cycle's obs positions equal the static pool, Regime B (series)
    reproduces Regime A to round-off for every cycle index (3D solve)."""
    grid, series, Xb, Yb, Y, rinv, npool, ens = _regimeB_setup(33)
    pool = series[0]
    same = [pool] * len(series)
    obs_idx = jnp.arange(npool)
    piA, pgA, _ = build_patch_geometry(grid, pool, localize_radius=2600.0)
    A = LETKF(system_dim=grid.shape[0], delta_t=0.01, ensemble_dim=ens,
              model_obj=None, grid_latlon=jnp.asarray(grid),
              obs_latlon=jnp.asarray(pool), localize_radius=2600.0,
              grid_chunk=None, patch_idx=piA, patch_gc=pgA)
    XaA = np.asarray(A._localized_analysis(Xb, Yb, Y, rinv, obs_idx, rho=1.0))
    pidx, pgc, _ = build_patch_geometry_series(
        grid, same, localize_radius=2600.0)
    B = LETKF(system_dim=grid.shape[0], delta_t=0.01, ensemble_dim=ens,
              model_obj=None, grid_latlon=jnp.asarray(grid),
              obs_latlon=jnp.asarray(pool), localize_radius=2600.0,
              grid_chunk=None, patch_idx_series=pidx, patch_gc_series=pgc,
              patch_pool_sizes=[npool] * len(same))
    for t in range(len(same)):
        XaB = np.asarray(B._localized_analysis(
            Xb, Yb, Y, rinv, obs_idx, rho=1.0, cycle_idx=t))
        assert np.allclose(XaB, XaA, rtol=0, atol=1e-10)


def test_letkf_patch_series_moving_obs_matches_per_cycle_static():
    """Moving observers: Regime B row t == a fresh Regime A build at cycle t's
    obs positions, and it is byte-identical under jit with a TRACED cycle_idx
    (the ``lax.scan`` consumption mode)."""
    grid, series, Xb, Yb, Y, rinv, npool, ens = _regimeB_setup(34)
    obs_idx = jnp.arange(npool)
    pidx, pgc, diag = build_patch_geometry_series(
        grid, series, localize_radius=2600.0)
    P = diag["P"]
    B = LETKF(system_dim=grid.shape[0], delta_t=0.01, ensemble_dim=ens,
              model_obj=None, grid_latlon=jnp.asarray(grid),
              obs_latlon=jnp.asarray(series[0]), localize_radius=2600.0,
              grid_chunk=None, patch_idx_series=pidx, patch_gc_series=pgc,
              patch_pool_sizes=[npool] * len(series))
    f = jax.jit(lambda t: B._localized_analysis(
        Xb, Yb, Y, rinv, obs_idx, rho=1.0, cycle_idx=t))
    for t, o in enumerate(series):
        pi, pw, _ = build_patch_geometry(
            grid, o, localize_radius=2600.0, patch_size=P)
        A = LETKF(system_dim=grid.shape[0], delta_t=0.01, ensemble_dim=ens,
                  model_obj=None, grid_latlon=jnp.asarray(grid),
                  obs_latlon=jnp.asarray(o), localize_radius=2600.0,
                  grid_chunk=None, patch_idx=pi, patch_gc=pw)
        XaA = np.asarray(A._localized_analysis(Xb, Yb, Y, rinv, obs_idx,
                                               rho=1.0))
        XaB = np.asarray(f(jnp.int32(t)))
        assert np.allclose(XaB, XaA, rtol=0, atol=1e-10)


def test_letkf_patch_series_window_stacked_4d():
    """Regime B window-stacks the per-cycle pool geometry across the 4D obs axis
    (n_obs = n_times * pool) exactly as the dense taper tiles; matches a fresh
    Regime A build at each cycle's positions."""
    grid, series, Xb, Yb0, Y0, rinv0, npool, ens = _regimeB_setup(35)
    n_times = 3
    rng = np.random.default_rng(350)
    nobs = n_times * npool
    Yb = jnp.asarray(rng.standard_normal((nobs, ens)))
    Y = jnp.asarray(rng.standard_normal(nobs))
    rinv = jnp.asarray(rng.uniform(0.5, 2.0, nobs))
    obs_idx = jnp.arange(nobs)
    pidx, pgc, diag = build_patch_geometry_series(
        grid, series, localize_radius=2600.0)
    P = diag["P"]
    B = LETKF(system_dim=grid.shape[0], delta_t=0.01, ensemble_dim=ens,
              model_obj=None, grid_latlon=jnp.asarray(grid),
              obs_latlon=jnp.asarray(series[0]), localize_radius=2600.0,
              grid_chunk=4, patch_idx_series=pidx, patch_gc_series=pgc,
              patch_pool_sizes=[npool] * len(series))
    for t, o in enumerate(series):
        pi, pw, _ = build_patch_geometry(
            grid, o, localize_radius=2600.0, patch_size=P)
        A = LETKF(system_dim=grid.shape[0], delta_t=0.01, ensemble_dim=ens,
                  model_obj=None, grid_latlon=jnp.asarray(grid),
                  obs_latlon=jnp.asarray(o), localize_radius=2600.0,
                  grid_chunk=4, patch_idx=pi, patch_gc=pw)
        XaA = np.asarray(A._localized_analysis(Xb, Yb, Y, rinv, obs_idx,
                                               rho=1.0))
        XaB = np.asarray(B._localized_analysis(
            Xb, Yb, Y, rinv, obs_idx, rho=1.0, cycle_idx=t))
        assert np.allclose(XaB, XaA, rtol=0, atol=1e-10)


def test_letkf_patch_callback_matches_series_and_regimeA():
    """The Regime-B host-callback fallback (per-cycle pure_callback rebuild)
    matches a fresh Regime A build at the same obs positions to round-off,
    including under jit (the pure_callback is scan-legal)."""
    rng = np.random.default_rng(36)
    G, npool, ens = 14, 10, 6
    grid = np.stack([rng.uniform(-70, 70, G), rng.uniform(0, 360, G)], axis=1)
    obs_grid_idx = rng.choice(G, size=npool, replace=False)
    o = grid[obs_grid_idx]                 # obs positions == grid rows
    obs_idx = jnp.arange(npool)
    Xb = jnp.asarray(rng.standard_normal((G, ens)))
    Yb = jnp.asarray(rng.standard_normal((npool, ens)))
    Y = jnp.asarray(rng.standard_normal(npool))
    rinv = jnp.asarray(rng.uniform(0.5, 2.0, npool))
    P = int(build_patch_geometry(grid, o, localize_radius=2600.0)[2]["P"])

    def cb(grid_ll, obs_ll, radius, units, Pfix):
        pi, pw, _ = build_patch_geometry(
            grid_ll, obs_ll, radius, units, patch_size=Pfix)
        return pi, pw

    C = LETKF(system_dim=G, delta_t=0.01, ensemble_dim=ens, model_obj=None,
              grid_latlon=jnp.asarray(grid), obs_latlon=jnp.asarray(o),
              localize_radius=2600.0, grid_chunk=4,
              patch_callback=cb, patch_callback_P=P)
    f = jax.jit(lambda: C._localized_analysis(
        Xb, Yb, Y, rinv, obs_idx, rho=1.0, obs_latlon_t=jnp.asarray(o)))
    XaC = np.asarray(f())
    pi, pw, _ = build_patch_geometry(
        grid, o, localize_radius=2600.0, patch_size=P)
    A = LETKF(system_dim=G, delta_t=0.01, ensemble_dim=ens, model_obj=None,
              grid_latlon=jnp.asarray(grid), obs_latlon=jnp.asarray(o),
              localize_radius=2600.0, grid_chunk=4, patch_idx=pi, patch_gc=pw)
    XaA = np.asarray(A._localized_analysis(Xb, Yb, Y, rinv, obs_idx, rho=1.0))
    assert np.allclose(XaC, XaA, rtol=0, atol=1e-10)


def test_letkf_patch_series_both_or_neither_and_mutual_exclusion():
    """series both-or-neither guard, mutual exclusion with static patch_idx, and
    the callback requiring patch_callback_P + grid_latlon."""
    G = 6
    with pytest.raises(ValueError, match="supplied together"):
        LETKF(system_dim=G, delta_t=0.01, ensemble_dim=4, model_obj=None,
              patch_idx_series=np.zeros((3, G, 2), np.int32))
    with pytest.raises(ValueError, match="not both"):
        LETKF(system_dim=G, delta_t=0.01, ensemble_dim=4, model_obj=None,
              patch_idx=np.zeros((G, 2), np.int32),
              patch_gc=np.zeros((G, 2)),
              patch_idx_series=np.zeros((3, G, 2), np.int32),
              patch_gc_series=np.zeros((3, G, 2)))
    with pytest.raises(ValueError, match="patch_callback_P"):
        LETKF(system_dim=G, delta_t=0.01, ensemble_dim=4, model_obj=None,
              grid_latlon=np.zeros((G, 2)), patch_callback=lambda *a: None)


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
