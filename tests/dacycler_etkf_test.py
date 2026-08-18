"""Tests for ETKF Data Assimilation Cycler (dabench.dacycler._etkf)"""

import pytest
import numpy as np
import jax
import jax.numpy as jnp
import jax.random as jrand
import dabench as dab
import dabench.dacycler._utils as dac_utils

jax.config.update("jax_enable_x64", True)

key = jrand.PRNGKey(42)


# ── Shared _obs_space_metrics reduction helper (unit test vs numpy) ────────
def test_obs_space_metrics_matches_numpy_reference():
    rng = np.random.default_rng(0)
    n_obs, ens = 6, 5
    y = rng.standard_normal(n_obs)
    Hxb = rng.standard_normal(n_obs)
    Hxa = rng.standard_normal(n_obs)
    active = np.array([1, 1, 0, 1, 0, 1], dtype=bool)
    sigma2 = np.array([0.25, 1.0, 4.0, 0.5, 9.0, 2.0])
    ens_obs = rng.standard_normal((n_obs, ens))

    out = dac_utils._obs_space_metrics(
        jnp.asarray(y), jnp.asarray(Hxb), jnp.asarray(Hxa),
        jnp.asarray(active), jnp.asarray(sigma2),
        ens_obs=jnp.asarray(ens_obs), return_per_obs=True)

    m = active.astype(float)
    n = m.sum()
    of = (y - Hxb)
    oa = (y - Hxa)
    assert float(out["n_active_obs"]) == pytest.approx(n)
    assert float(out["o_minus_f_rms"]) == pytest.approx(
        np.sqrt(np.sum((of * m) ** 2) / n))
    assert float(out["o_minus_a_rms"]) == pytest.approx(
        np.sqrt(np.sum((oa * m) ** 2) / n))
    assert float(out["bias_f"]) == pytest.approx(np.sum(of * m) / n)
    assert float(out["bias_a"]) == pytest.approx(np.sum(oa * m) / n)
    per_obs_var = np.mean(ens_obs ** 2, axis=1)
    assert float(out["obs_space_spread_background"]) == pytest.approx(
        np.sqrt(np.sum(per_obs_var * m) / n))
    assert float(out["sigma_obs_max"]) == pytest.approx(
        np.sqrt(np.max(sigma2[active])))
    # Per-obs debug arrays: masked entries are NaN, active match innovations.
    of_full = np.asarray(out["o_minus_f"])
    assert np.all(np.isnan(of_full[~active]))
    assert np.allclose(of_full[active], of[active])
    assert np.allclose(np.asarray(out["obs_active"]), m)


def test_obs_space_metrics_zero_active_is_nan():
    y = jnp.asarray(np.zeros(4))
    active = jnp.asarray(np.zeros(4, dtype=bool))
    sigma2 = jnp.asarray(np.ones(4))
    out = dac_utils._obs_space_metrics(y, y, y, active, sigma2)
    assert float(out["n_active_obs"]) == 0.0
    for k in ("o_minus_f_rms", "o_minus_a_rms", "bias_f", "bias_a",
              "obs_space_spread_background", "sigma_obs_max"):
        assert bool(np.isnan(float(out[k])))


def test_obs_space_metrics_spread_nan_when_no_ensemble():
    y = jnp.asarray(np.ones(3))
    active = jnp.asarray(np.ones(3, dtype=bool))
    sigma2 = jnp.asarray(np.ones(3))
    out = dac_utils._obs_space_metrics(y, y, y, active, sigma2, ens_obs=None)
    assert bool(np.isnan(float(out["obs_space_spread_background"])))


@pytest.fixture
def l96_nature_run():
    """Defines class Lorenz96 object for rest of tests."""
    l96 = dab.data.Lorenz96(system_dim=5, store_as_jax=True, delta_t=0.01)
    return l96.generate(n_steps=25)

@pytest.fixture
def obs_vec_l96(l96_nature_run):
    """Generate observations for rest of tests."""
    obs_l96 = dab.observer.Observer(
        l96_nature_run,
        times=l96_nature_run['time'].data[np.arange(0, 25, 5)],
        random_location_count=3,
        error_bias=0.1,
        error_sd=1.0,
        random_seed=91,
        stationary_observers=True,
        store_as_jax=True
    )

    return obs_l96.observe()

@pytest.fixture
def l96_fc_model():
    model_l96 = dab.data.Lorenz96(system_dim=5, store_as_jax=True, delta_t=0.05)

    class L96Model(dab.model.Model):                                                                       
        """Defines model wrapper for Lorenz96 to test forecasting."""
        def forecast(self, state_vec, n_steps):
            new_vec = self.model_obj.generate(x0=state_vec['x'].data, n_steps=n_steps)

            return new_vec.isel(time=-1).assign_attrs(delta_t=0.01), new_vec

    return L96Model(model_obj=model_l96)

@pytest.fixture
def etkf_cycler(l96_fc_model):
    dc = dab.dacycler.ETKF(
        system_dim=5,
        delta_t=0.01,
        ensemble_dim=8,
        model_obj=l96_fc_model)
    
    return dc

def test_etkf_l96(l96_nature_run, obs_vec_l96, etkf_cycler):
    cur_tstep=10
    init_noise = jrand.normal(key, shape=(8, 5))
    init_state = l96_nature_run.isel(time=cur_tstep)
    init_state = init_state.assign(
        x=(['ensemble','index'], init_state['x'].data + init_noise)
    )
    start_time = init_state['time'].data

    out_sv = etkf_cycler.cycle(
        input_state=init_state,
        start_time=start_time,
        obs_vector=obs_vec_l96,
        obs_error_sd=1.5,
        analysis_window=0.1,
        n_cycles=10,
        return_forecast=True
        )
    out_sv = out_sv.stack(time=['cycle','cycle_timestep']).transpose('time',...)


    out_sv_mean = out_sv.mean(dim='ensemble')

    assert out_sv['x'].shape == (100, 8, 5)
    assert out_sv_mean['x'].shape == (100, 5)
    # Check that ensemble members are different
    assert not jnp.allclose(
        out_sv['x'].values[-1, 1, :],
        out_sv['x'].values[-1, 0, :],
    )
    # Check first cycle against presaved results
    assert jnp.allclose(
        out_sv['x'].values[0, 0, :],
        jnp.array([-0.85402591, 1.03480315, 0.51005132, 6.61546551, 8.1166806])
    )
    # Check last cycle against presaved results
    assert jnp.allclose(
        out_sv['x'].values[-1, 0, :],
        jnp.array([0.66697948, 3.15465627, 5.39288975, -4.96130847, 2.17202611])
    )
    # Check mean against presaved results
    assert jnp.allclose(
        out_sv_mean['x'].values[-1, :],
        jnp.array([1.45024252, 3.81627191, 5.4507981, 1.21646539, 0.09439264])
    )


def _etkf_init(l96_nature_run):
    init_noise = jrand.normal(key, shape=(8, 5))
    init_state = l96_nature_run.isel(time=10)
    return init_state.assign(
        x=(['ensemble', 'index'], init_state['x'].data + init_noise))


def test_etkf_obs_metrics(l96_nature_run, obs_vec_l96, etkf_cycler):
    """return_metrics=True yields the 7 per-cycle obs-space scalars, keeps the
    analysis byte-identical to the default path, and honors masking/debug."""
    init_state = _etkf_init(l96_nature_run)
    kw = dict(input_state=init_state, start_time=init_state['time'].data,
              obs_vector=obs_vec_l96, obs_error_sd=1.5, analysis_window=0.1,
              n_cycles=10, return_forecast=True)

    ana_default = etkf_cycler.cycle(**kw)
    ana, metrics = etkf_cycler.cycle(return_metrics=True, **kw)
    # (1) Byte-identical analysis regardless of return_metrics.
    assert np.array_equal(np.asarray(ana_default['x'].data),
                          np.asarray(ana['x'].data))

    # (2) The 7 aggregate scalars, one value per cycle.
    for var in ("o_minus_f_rms", "o_minus_a_rms", "bias_f", "bias_a",
                "obs_space_spread_background", "sigma_obs_max",
                "n_active_obs"):
        assert metrics[var].shape == (10,)
    # Restrict finite-value assertions to cycles that actually saw obs.
    n_active = np.asarray(metrics["n_active_obs"].data)
    act = n_active > 0
    assert bool(np.any(act))
    # (3) Zero-active cycles -> n_active_obs==0 and NaN scalars.
    assert bool(np.all(n_active[~act] == 0))
    assert bool(np.all(np.isnan(np.asarray(
        metrics["o_minus_f_rms"].data)[~act])))
    # (4) Ensemble spread is finite / positive on active cycles (EnKF).
    spread = np.asarray(metrics["obs_space_spread_background"].data)
    assert bool(np.all(np.isfinite(spread[act])))
    assert bool(np.all(spread[act] > 0))
    # sigma_obs_max = obs_error_sd on active cycles.
    assert np.allclose(np.asarray(metrics["sigma_obs_max"].data)[act], 1.5)
    # (7) O-A RMS <= O-F RMS on well-observed cycles (analysis pulls to obs).
    of = np.asarray(metrics["o_minus_f_rms"].data)[act]
    oa = np.asarray(metrics["o_minus_a_rms"].data)[act]
    assert bool(np.all(oa <= of + 1e-8))

    # (6) debug adds the 3 per-obs arrays; masked slots -> NaN / 0.
    _, metrics_dbg = etkf_cycler.cycle(
        return_metrics=True, metrics_mode="debug", **kw)
    for var in ("o_minus_f", "o_minus_a", "obs_active"):
        assert metrics_dbg[var].dims == ("cycle", "obs")
        assert metrics_dbg[var].shape[0] == 10
    active = np.asarray(metrics_dbg["obs_active"].data).astype(bool)
    of_full = np.asarray(metrics_dbg["o_minus_f"].data)
    assert np.all(np.isnan(of_full[~active]))
    assert bool(np.all(np.isfinite(of_full[active])))

    # (8) Differentiable-by-default: metric leaves are JAX arrays unless
    # detach_metrics=True, which numpy-converts with NaN-aware value parity
    # (the eval/forensics path opts in to detach to preserve host behaviour).
    assert isinstance(metrics["o_minus_f_rms"].data, jax.Array)
    _, metrics_np = etkf_cycler.cycle(
        return_metrics=True, detach_metrics=True, **kw)
    assert isinstance(metrics_np["o_minus_f_rms"].data, np.ndarray)
    for var in ("o_minus_f_rms", "o_minus_a_rms", "bias_f", "bias_a",
                "obs_space_spread_background", "sigma_obs_max",
                "n_active_obs"):
        assert np.allclose(np.asarray(metrics[var].data),
                           np.asarray(metrics_np[var].data), equal_nan=True)

    # (9) The differentiable zero-masked innovations equal the NaN-masked
    # eval fields on ACTIVE obs and are finite (0) on inactive ones.
    for masked, nan_field in (("o_minus_f_masked", "o_minus_f"),
                              ("o_minus_a_masked", "o_minus_a")):
        fm = np.asarray(metrics_dbg[masked].data)
        fn = np.asarray(metrics_dbg[nan_field].data)
        assert metrics_dbg[masked].dims == ("cycle", "obs")
        assert bool(np.all(np.isfinite(fm)))
        assert np.allclose(fm[active], fn[active])
        assert bool(np.all(fm[~active] == 0))


def test_etkf_obs_metrics_bad_mode(l96_nature_run, obs_vec_l96, etkf_cycler):
    init_state = _etkf_init(l96_nature_run)
    with pytest.raises(ValueError):
        etkf_cycler.cycle(
            input_state=init_state, start_time=init_state['time'].data,
            obs_vector=obs_vec_l96, obs_error_sd=1.5, analysis_window=0.1,
            n_cycles=10, return_forecast=True,
            return_metrics=True, metrics_mode="bogus")


# ── 3D-FGAT ETKF cycle (global, non-localized) ──────────────────────────────
# A longer nature run + fully-observable network so the cycle test validates
# the FGAT machinery (innovations at each obs's true time, increment at tau)
# rather than the observability boundary.  Mirrors the LETKF-FGAT cycle test.
@pytest.fixture
def fgat_nature():
    l96 = dab.data.Lorenz96(system_dim=5, store_as_jax=True, delta_t=0.01)
    return l96.generate(n_steps=120)


@pytest.fixture
def fgat_fc_model():
    m = dab.data.Lorenz96(system_dim=5, store_as_jax=True, delta_t=0.01)

    class L96Model(dab.model.Model):
        def forecast(self, state_vec, n_steps):
            v = self.model_obj.generate(x0=state_vec['x'].data, n_steps=n_steps)
            return v.isel(time=-1).assign_attrs(delta_t=0.01), v

    return L96Model(model_obj=m)


@pytest.fixture
def fgat_obs(fgat_nature):
    """Fully-observable (5/5) obs every 0.05 -> multiple obs times per 0.1
    window (the multi-time regime that breaks a plain 3D filter)."""
    return dab.observer.Observer(
        fgat_nature, times=fgat_nature['time'].data[np.arange(0, 120, 5)],
        random_location_count=5, error_bias=0.0, error_sd=0.5,
        random_seed=91, stationary_observers=True, store_as_jax=True).observe()


def _fgat_ens_init(nature, ens=10):
    init_noise = jrand.normal(key, shape=(ens, 5))
    init_state = nature.isel(time=10)
    return init_state.assign(
        x=(['ensemble', 'index'], init_state['x'].data + init_noise))


def _fgat_run(cycler, init_state, obs, obs_error_sd=1.0):
    return cycler.cycle(
        input_state=init_state, start_time=init_state['time'].data,
        obs_vector=obs, obs_error_sd=obs_error_sd, analysis_window=0.1,
        n_cycles=10, return_forecast=True)


def _fgat_ens_rmse(out, nature):
    """Ensemble-mean analysis RMSE vs nature at each cycle's incoming IC
    (``cycle_timestep=0`` is the window-end analysis, regardless of tau)."""
    ana = np.asarray(out.isel(cycle_timestep=0).mean('ensemble')['x'].data)
    nat = np.asarray(nature['x'].data)
    idx = [10 + c * 10 for c in range(ana.shape[0])]
    return float(np.sqrt(np.mean((ana - nat[idx]) ** 2)))


def test_etkf_fgat_l96_cycle(fgat_nature, fgat_obs, fgat_fc_model):
    """Global 3D-FGAT ETKF over multi-obs-time windows: valid shapes, finite,
    distinct members, and DA skill vs a near no-op R.  A plain 3D ETKF collapses
    all in-window obs onto one static state; FGAT forms innovations at each
    obs's true time while building the increment at the single analysis time."""
    ens = 10
    init_state = _fgat_ens_init(fgat_nature, ens=ens)
    etkf = dab.dacycler.ETKF(system_dim=5, delta_t=0.01, ensemble_dim=ens,
                             model_obj=fgat_fc_model, fgat=True)
    out = _fgat_run(etkf, init_state, fgat_obs)

    assert out['x'].shape == (10, ens, 10, 5)
    assert bool(np.all(np.isfinite(np.asarray(out['x'].data))))
    # Transform keeps ensemble members distinct.
    assert not jnp.allclose(out['x'].values[-1, 1, 0, :],
                            out['x'].values[-1, 0, 0, :])
    # FGAT DA pulls the analysis toward truth vs a near no-op R.
    rmse_da = _fgat_ens_rmse(out, fgat_nature)
    etkf_noop = dab.dacycler.ETKF(system_dim=5, delta_t=0.01, ensemble_dim=ens,
                                  model_obj=fgat_fc_model, fgat=True)
    rmse_noop = _fgat_ens_rmse(
        _fgat_run(etkf_noop, init_state, fgat_obs, obs_error_sd=1.0e6),
        fgat_nature)
    assert rmse_da < rmse_noop


# ── analysis_time_index placement sweep (start / mid / end) ──────────────────
@pytest.mark.parametrize("cls,kw", [
    (dab.dacycler.ETKF, {}),
    (dab.dacycler.LETKF, {"localize_radius": 1.5}),
])
@pytest.mark.parametrize("ati", ["start", "mid", "end"])
def test_ensemble_fgat_analysis_time_placement(
        fgat_nature, fgat_obs, fgat_fc_model, cls, kw, ati):
    """3D-FGAT ETKF/LETKF must yield a finite, error-reducing analysis for the
    analysis time placed at the window START, MIDDLE, and END.  The IC handoff
    always re-forecasts tau -> window end, so ``cycle_timestep=0`` is the
    window-end analysis for every placement."""
    ens = 10
    da = cls(system_dim=5, delta_t=0.01, ensemble_dim=ens,
             model_obj=fgat_fc_model, fgat=True,
             analysis_time_index=ati, **kw)
    out = _fgat_run(da, _fgat_ens_init(fgat_nature, ens=ens), fgat_obs)
    assert out['x'].shape == (10, ens, 10, 5)
    assert bool(np.all(np.isfinite(np.asarray(out['x'].data))))

    noop = cls(system_dim=5, delta_t=0.01, ensemble_dim=ens,
               model_obj=fgat_fc_model, fgat=True,
               analysis_time_index=ati, **kw)
    rmse_da = _fgat_ens_rmse(out, fgat_nature)
    rmse_noop = _fgat_ens_rmse(
        _fgat_run(noop, _fgat_ens_init(fgat_nature, ens=ens), fgat_obs,
                  obs_error_sd=1.0e6),
        fgat_nature)
    assert rmse_da < rmse_noop


def test_etkf_fgat_bad_analysis_time_index_raises(fgat_fc_model):
    with pytest.raises(ValueError, match="analysis_time_index"):
        c = dab.dacycler.ETKF(system_dim=5, delta_t=0.01, ensemble_dim=10,
                              model_obj=fgat_fc_model, fgat=True,
                              analysis_time_index="bogus")
        c.steps_per_window = 11
        c._resolve_analysis_index()
