"""Tests for 4D-ETKF Data Assimilation Cycler (dabench.dacycler._etkf4d)"""

import pytest
import numpy as np
import jax.numpy as jnp
import jax.random as jrand
import dabench as dab
from dabench.dacycler import ETKF4D


key = jrand.PRNGKey(42)


@pytest.fixture
def l96_nature_run():
    """Long Lorenz96 nature run so the cycling window fits."""
    l96 = dab.data.Lorenz96(system_dim=5, store_as_jax=True, delta_t=0.01)
    return l96.generate(n_steps=120)


@pytest.fixture
def obs_vec_l96(l96_nature_run):
    """Observations distributed across the window (every 5 model steps)."""
    obs_l96 = dab.observer.Observer(
        l96_nature_run,
        times=l96_nature_run['time'].data[np.arange(0, 120, 5)],
        random_location_count=3,
        error_bias=0.0,
        error_sd=1.0,
        random_seed=91,
        stationary_observers=True,
        store_as_jax=True
    )

    return obs_l96.observe()


@pytest.fixture
def l96_fc_model():
    # delta_t matches the cycler so in-window obs times align with model steps.
    model_l96 = dab.data.Lorenz96(system_dim=5, store_as_jax=True, delta_t=0.01)

    class L96Model(dab.model.Model):
        """Defines model wrapper for Lorenz96 to test forecasting."""
        def forecast(self, state_vec, n_steps):
            new_vec = self.model_obj.generate(
                x0=state_vec['x'].data, n_steps=n_steps)

            return new_vec.isel(time=-1).assign_attrs(delta_t=0.01), new_vec

    return L96Model(model_obj=model_l96)


@pytest.fixture
def etkf4d_cycler(l96_fc_model):
    return ETKF4D(
        system_dim=5,
        delta_t=0.01,
        ensemble_dim=8,
        model_obj=l96_fc_model)


def _make_init(l96_nature_run, cur_tstep=10):
    init_noise = jrand.normal(key, shape=(8, 5))
    init_state = l96_nature_run.isel(time=cur_tstep)
    init_state = init_state.assign(
        x=(['ensemble', 'index'], init_state['x'].data + init_noise)
    )
    return init_state


def test_etkf4d_flags():
    """4D-ETKF must ride the _in_4d ensemble plumbing."""
    assert ETKF4D._in_4d is True
    assert ETKF4D._uses_ensemble is True


def test_etkf4d_l96_shapes(l96_nature_run, obs_vec_l96, etkf4d_cycler):
    """cycle() runs 4D and returns an ensemble analysis of the right shape."""
    init_state = _make_init(l96_nature_run)

    out_sv = etkf4d_cycler.cycle(
        input_state=init_state,
        start_time=init_state['time'].data,
        obs_vector=obs_vec_l96,
        obs_error_sd=1.0,
        analysis_window=0.1,
        n_cycles=10,
        return_forecast=True
    )

    assert out_sv['x'].shape == (10, 8, 10, 5)
    out_stacked = out_sv.stack(
        time=['cycle', 'cycle_timestep']).transpose('time', ...)
    assert out_stacked['x'].shape == (100, 8, 5)
    assert bool(np.all(np.isfinite(np.asarray(out_sv['x'].data))))
    # Ensemble members remain distinct after the transform.
    assert not jnp.allclose(
        out_sv['x'].values[-1, 1, 0, :],
        out_sv['x'].values[-1, 0, 0, :],
    )


def test_etkf4d_reduces_rmse(l96_nature_run, obs_vec_l96, etkf4d_cycler):
    """Window-distributed innovations must pull the analysis toward truth."""
    init_state = _make_init(l96_nature_run)

    def _run(obs_error_sd):
        return etkf4d_cycler.cycle(
            input_state=init_state,
            start_time=init_state['time'].data,
            obs_vector=obs_vec_l96,
            obs_error_sd=obs_error_sd,
            analysis_window=0.1,
            n_cycles=10,
            return_forecast=True
        )

    def _rmse(out):
        # cycle_timestep=0 is each cycle's incoming IC -- the previous cycle's
        # window-end analysis (default filter placement) -- ensemble-mean vs
        # nature (10 model steps/cycle, init at nature index 10).
        ana = np.asarray(out.isel(cycle_timestep=0).mean('ensemble')['x'].data)
        nat = np.asarray(l96_nature_run['x'].data)
        idx = [10 + c * 10 for c in range(ana.shape[0])]
        return float(np.sqrt(np.mean((ana - nat[idx]) ** 2)))

    rmse_da = _rmse(_run(1.0))
    rmse_noop = _rmse(_run(1.0e6))  # huge R -> near no-op update
    assert rmse_da < rmse_noop


def test_etkf4d_apply_additive_structured(etkf4d_cycler):
    """Additive inflation is mean-zero, RMS-calibrated, and lives in the
    forecast-difference subspace (col-span of the prior perturbations)."""
    c = etkf4d_cycler
    rng = np.random.default_rng(0)
    Xb_pert = jnp.asarray(rng.standard_normal((12, 8)))
    Xb_pert = Xb_pert - jnp.mean(Xb_pert, axis=1, keepdims=True)
    Xa_pert = jnp.zeros_like(Xb_pert)

    c.additive_inflation = 0.05
    E = c._apply_additive(Xb_pert, Xa_pert, jrand.PRNGKey(0))
    # Mean-zero across members (does not shift the analysis mean).
    assert np.allclose(np.asarray(E).mean(axis=1), 0.0, atol=1e-6)
    # Rescaled to the requested absolute per-element RMS.
    assert abs(float(jnp.sqrt(jnp.mean(E ** 2))) - 0.05) < 1e-6
    # In col-span(Xb_pert): the orthogonal-complement residual is ~0.
    Q, _ = np.linalg.qr(np.asarray(Xb_pert))
    resid = np.asarray(E) - Q @ (Q.T @ np.asarray(E))
    assert np.linalg.norm(resid) < 1e-5

    # additive_inflation = 0 is an exact no-op.
    c.additive_inflation = 0.0
    E0 = c._apply_additive(Xb_pert, Xa_pert, jrand.PRNGKey(0))
    assert np.allclose(np.asarray(E0), np.asarray(Xa_pert))


def test_etkf4d_additive_lifts_spread(l96_nature_run, obs_vec_l96, l96_fc_model):
    """Structured additive inflation raises the analysis-ensemble spread (the
    absolute floor multiplicative levers cannot set) and stays finite."""
    init_state = _make_init(l96_nature_run)

    def _run(additive):
        cycler = ETKF4D(system_dim=5, delta_t=0.01, ensemble_dim=8,
                        model_obj=l96_fc_model,
                        additive_inflation=additive, additive_seed=0)
        return cycler.cycle(
            input_state=init_state,
            start_time=init_state['time'].data,
            obs_vector=obs_vec_l96,
            obs_error_sd=1.0,
            analysis_window=0.1,
            n_cycles=10,
            return_forecast=True
        )

    def _spread(out):
        var = out.isel(cycle_timestep=0).var('ensemble', ddof=1)['x'].data
        return float(np.sqrt(np.asarray(var).mean()))

    out0 = _run(0.0)
    out1 = _run(0.5)
    assert bool(np.all(np.isfinite(np.asarray(out1['x'].data))))
    assert _spread(out1) > _spread(out0)


def test_etkf4d_obs_metrics(l96_nature_run, obs_vec_l96, etkf4d_cycler,
                            l96_fc_model):
    """4D obs-space metrics: byte-identical analysis, 7 per-cycle scalars,
    finite ensemble spread, O-A<=O-F (time-matched score), debug per-obs arrays.

    The default O-A score is CAUSAL (obs before the analysis time tau are
    scored against the tau-time analysis, no back-propagation), so with the
    default placement (tau=window end) O-A need NOT be <= O-F.  The
    ``oa_score_mode="time_matched"`` score places a diagnostic analysis at the
    window start and re-forecasts it, restoring the O-A <= O-F sanity property.
    """
    init_state = _make_init(l96_nature_run)
    kw = dict(input_state=init_state, start_time=init_state['time'].data,
              obs_vector=obs_vec_l96, obs_error_sd=1.0, analysis_window=0.1,
              n_cycles=10, return_forecast=True)

    ana_default = etkf4d_cycler.cycle(**kw)
    ana, metrics = etkf4d_cycler.cycle(return_metrics=True, **kw)
    assert np.array_equal(np.asarray(ana_default['x'].data),
                          np.asarray(ana['x'].data))

    for var in ("o_minus_f_rms", "o_minus_a_rms", "bias_f", "bias_a",
                "obs_space_spread_background", "sigma_obs_max",
                "n_active_obs"):
        assert metrics[var].shape == (10,)
    n_active = np.asarray(metrics["n_active_obs"].data)
    act = n_active > 0
    assert bool(np.any(act))
    spread = np.asarray(metrics["obs_space_spread_background"].data)
    assert bool(np.all(np.isfinite(spread[act])))
    assert bool(np.all(spread[act] > 0))
    assert np.allclose(np.asarray(metrics["sigma_obs_max"].data)[act], 1.0)
    # Causal O-A is finite (the default score); no O-A<=O-F guarantee.
    assert bool(np.all(np.isfinite(
        np.asarray(metrics["o_minus_a_rms"].data)[act])))

    # Time-matched O-A restores the O-A <= O-F sanity property; the analysis is
    # byte-identical (oa_score_mode is a metrics-only knob).
    tm = ETKF4D(system_dim=5, delta_t=0.01, ensemble_dim=8,
                model_obj=l96_fc_model, oa_score_mode="time_matched")
    ana_tm, metrics_tm = tm.cycle(return_metrics=True, **kw)
    assert np.array_equal(np.asarray(ana_default['x'].data),
                          np.asarray(ana_tm['x'].data))
    of = np.asarray(metrics_tm["o_minus_f_rms"].data)[act]
    oa = np.asarray(metrics_tm["o_minus_a_rms"].data)[act]
    assert bool(np.all(oa <= of + 1e-8))

    _, metrics_dbg = etkf4d_cycler.cycle(
        return_metrics=True, metrics_mode="debug", **kw)
    for var in ("o_minus_f", "o_minus_a", "obs_active"):
        assert metrics_dbg[var].dims == ("cycle", "obs")
        assert metrics_dbg[var].shape[0] == 10


def test_etkf4d_apply_rtpp(etkf4d_cycler):
    """RTPP linearly blends analysis toward the prior perturbations:
    alpha=0 is an exact no-op, alpha=1 returns the prior perturbations, and
    intermediate alpha is the exact convex combination."""
    c = etkf4d_cycler
    rng = np.random.default_rng(0)
    Xb_pert = jnp.asarray(rng.standard_normal((5, 8)))
    Xa_pert = jnp.asarray(rng.standard_normal((5, 8)))

    c.rtpp_relaxation = 0.0
    assert np.allclose(np.asarray(c._apply_rtpp(Xb_pert, Xa_pert)),
                       np.asarray(Xa_pert))
    c.rtpp_relaxation = 1.0
    assert np.allclose(np.asarray(c._apply_rtpp(Xb_pert, Xa_pert)),
                       np.asarray(Xb_pert))
    c.rtpp_relaxation = 0.3
    exp = 0.7 * np.asarray(Xa_pert) + 0.3 * np.asarray(Xb_pert)
    assert np.allclose(np.asarray(c._apply_rtpp(Xb_pert, Xa_pert)), exp)


def test_etkf4d_rtpp_lifts_spread(l96_nature_run, obs_vec_l96, l96_fc_model):
    """RTPP raises the analysis-ensemble spread relative to no relaxation and
    stays finite (re-injects the prior perturbation structure)."""
    init_state = _make_init(l96_nature_run)

    def _run(rtpp):
        cycler = ETKF4D(system_dim=5, delta_t=0.01, ensemble_dim=8,
                        model_obj=l96_fc_model, rtpp_relaxation=rtpp)
        return cycler.cycle(
            input_state=init_state,
            start_time=init_state['time'].data,
            obs_vector=obs_vec_l96,
            obs_error_sd=1.0,
            analysis_window=0.1,
            n_cycles=10,
            return_forecast=True
        )

    def _spread(out):
        var = out.isel(cycle_timestep=0).var('ensemble', ddof=1)['x'].data
        return float(np.sqrt(np.asarray(var).mean()))

    out0 = _run(0.0)
    out1 = _run(0.8)
    assert bool(np.all(np.isfinite(np.asarray(out1['x'].data))))
    assert _spread(out1) > _spread(out0)


# ── analysis_time_index placement sweep (start / mid / end) ──────────────────
@pytest.mark.parametrize("cls,kw", [
    (dab.dacycler.ETKF4D, {}),
    (dab.dacycler.LETKF4D, {"localize_radius": 1.5}),
])
@pytest.mark.parametrize("ati", ["start", "mid", "end"])
def test_etkf4d_analysis_time_placement(
        l96_nature_run, obs_vec_l96, l96_fc_model, cls, kw, ati):
    """The 4D ensemble filters must yield a finite, error-reducing analysis for
    the transform-weight time placed at the window START, MIDDLE, and END.  The
    IC handoff always re-forecasts tau -> window end, so ``cycle_timestep=0`` is
    the window-end analysis for every placement."""
    init_state = _make_init(l96_nature_run)

    def _run(obs_error_sd):
        cycler = cls(system_dim=5, delta_t=0.01, ensemble_dim=8,
                     model_obj=l96_fc_model, analysis_time_index=ati, **kw)
        return cycler.cycle(
            input_state=init_state, start_time=init_state['time'].data,
            obs_vector=obs_vec_l96, obs_error_sd=obs_error_sd,
            analysis_window=0.1, n_cycles=10, return_forecast=True)

    def _rmse(out):
        ana = np.asarray(out.isel(cycle_timestep=0).mean('ensemble')['x'].data)
        nat = np.asarray(l96_nature_run['x'].data)
        idx = [10 + c * 10 for c in range(ana.shape[0])]
        return float(np.sqrt(np.mean((ana - nat[idx]) ** 2)))

    out_da = _run(1.0)
    assert bool(np.all(np.isfinite(np.asarray(out_da['x'].data))))
    rmse_da = _rmse(out_da)
    rmse_noop = _rmse(_run(1.0e6))     # huge R -> near no-op update
    assert rmse_da < rmse_noop


def _batched_cycle(cycler, init_state, obs_vector, obs_error_sd, start_time0,
                   analysis_window, n_cycles, batch, return_metrics=False,
                   metrics_mode="default"):
    """Reference batched driver: split one cycle() into sequential chunks.

    Mirrors the MLTLM run_4dvar._run_cycle_batched contract: each batch passes
    start_time advanced by cycles_done*analysis_window (obs are gathered by
    ABSOLUTE time), threads the EXACT scan carry via return_final_state=True as
    the next batch's input_state, and concatenates the per-batch outputs along
    'cycle'.  Returns the SAME (analysis[, metrics]) shape as a monolithic call.
    """
    import xarray as xr
    cur = init_state
    a_parts, m_parts = [], []
    done = 0
    while done < n_cycles:
        this_len = min(batch, n_cycles - done)
        out = cycler.cycle(
            input_state=cur,
            start_time=start_time0 + done * analysis_window,
            obs_vector=obs_vector, obs_error_sd=obs_error_sd,
            analysis_window=analysis_window, n_cycles=this_len,
            return_forecast=True, return_final_state=True,
            return_metrics=return_metrics, metrics_mode=metrics_mode)
        if return_metrics:
            a_ds, m_ds, final_state = out
            m_parts.append(m_ds.to_dataset()
                           if hasattr(m_ds, "to_dataset") else m_ds)
        else:
            a_ds, final_state = out
        a_parts.append(a_ds)
        cur = final_state
        done += this_len
    analysis = xr.concat(a_parts, dim="cycle")
    if return_metrics:
        return analysis, xr.concat(m_parts, dim="cycle")
    return analysis


def test_etkf4d_return_final_state_is_dropped_boundary_frame(
        l96_nature_run, obs_vec_l96, etkf4d_cycler):
    """return_final_state yields the EXACT frame return_forecast drops.

    With return_forecast=True the emitted output drops cycle_timestep=-1 (the
    window-boundary background the scan carries into the next cycle).  A single
    cycle() over N cycles must return, as its final_state, the same tensor a
    (N+1)-cycle run would expose as its N-th cycle's dropped boundary frame --
    i.e. running N cycles then 1 more from that final_state must reproduce the
    monolithic (N+1)-cycle output byte-for-byte.
    """
    init_state = _make_init(l96_nature_run)
    kw = dict(obs_vector=obs_vec_l96, obs_error_sd=1.0, analysis_window=0.1)
    t0 = init_state['time'].data

    # n_cycles=8 with batch=4 -> chunks [4, 4].  (The _get_all_times arange
    # float-endpoint bug that used to corrupt n in {3,6,12} at window=0.1 is now
    # fixed in dabench.dacycler._utils; see test_etkf4d_batched_equals_
    # monolithic_previously_buggy_n for the direct regression on n=6.)
    mono = etkf4d_cycler.cycle(
        input_state=init_state, start_time=t0, n_cycles=8,
        return_forecast=True, **kw)
    batched = _batched_cycle(
        etkf4d_cycler, init_state, obs_vec_l96, 1.0, t0, 0.1, 8, batch=4)

    a_mono = np.asarray(mono['x'].data)
    a_bat = np.asarray(batched['x'].data)
    assert a_mono.shape == a_bat.shape
    assert np.array_equal(a_mono, a_bat), (
        "batched analysis must be byte-identical to the monolithic scan")


@pytest.mark.parametrize("batch", [1, 2, 4, 8])
def test_etkf4d_batched_equals_monolithic(
        l96_nature_run, obs_vec_l96, etkf4d_cycler, batch):
    """Batched cycling == monolithic cycling (analysis + metrics), any batch.

    The whole point of the batched driver is progress/memory WITHOUT changing
    the numeric result: for every batch size the concatenated analysis and the
    per-cycle obs metrics must match a single 8-cycle cycle() to round-off
    (byte-identical, since the handoff threads the exact scan carry).  Batch
    sizes 1/2/4/8 decompose 8 cycles into chunks {1,2,4,8}, so the test isolates
    the batching handoff.
    """
    init_state = _make_init(l96_nature_run)
    t0 = init_state['time'].data
    a_mono, m_mono = etkf4d_cycler.cycle(
        input_state=init_state, start_time=t0, obs_vector=obs_vec_l96,
        obs_error_sd=1.0, analysis_window=0.1, n_cycles=8,
        return_forecast=True, return_metrics=True)
    a_bat, m_bat = _batched_cycle(
        etkf4d_cycler, init_state, obs_vec_l96, 1.0, t0, 0.1, 8, batch=batch,
        return_metrics=True)

    assert np.array_equal(np.asarray(a_mono['x'].data),
                          np.asarray(a_bat['x'].data))
    m_mono_ds = m_mono.to_dataset() if hasattr(m_mono, "to_dataset") else m_mono
    for var in ("o_minus_f_rms", "o_minus_a_rms", "bias_f", "bias_a",
                "obs_space_spread_background", "sigma_obs_max",
                "n_active_obs"):
        assert np.allclose(np.asarray(m_mono_ds[var].data),
                           np.asarray(m_bat[var].data), rtol=0, atol=0), var


@pytest.mark.parametrize("n_cycles", [3, 6])
def test_etkf4d_previously_buggy_n_runs_and_batches(
        l96_nature_run, obs_vec_l96, etkf4d_cycler, n_cycles):
    """n in {3,6} at window=0.1 now runs (regression for the arange bug).

    Before the _get_all_times fix, ``jnp.arange(0, n*0.1, 0.1)`` returned n+1
    times for n in {3,6,12}, so a plain ``cycle(n_cycles=n, analysis_window=0.1)``
    either raised a broadcast error or silently shifted the obs schedule.  This
    pins that these counts now (a) run, (b) return exactly n cycles, and (c)
    stay byte-identical between monolithic and batched execution.
    """
    init_state = _make_init(l96_nature_run)
    t0 = init_state['time'].data
    mono = etkf4d_cycler.cycle(
        input_state=init_state, start_time=t0, obs_vector=obs_vec_l96,
        obs_error_sd=1.0, analysis_window=0.1, n_cycles=n_cycles,
        return_forecast=True)
    assert int(mono.sizes['cycle']) == n_cycles
    assert bool(np.all(np.isfinite(np.asarray(mono['x'].data))))
    batched = _batched_cycle(
        etkf4d_cycler, init_state, obs_vec_l96, 1.0, t0, 0.1, n_cycles, batch=2)
    assert np.array_equal(np.asarray(mono['x'].data),
                          np.asarray(batched['x'].data))


def test_etkf4d_bad_analysis_time_index_raises(l96_fc_model):
    with pytest.raises(ValueError, match="analysis_time_index"):
        c = ETKF4D(system_dim=5, delta_t=0.01, ensemble_dim=8,
                   model_obj=l96_fc_model, analysis_time_index="bogus")
        c.steps_per_window = 11
        c._resolve_analysis_index()
