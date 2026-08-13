"""Tests for Var4D and Var4D-Backprop Data Assimilation Cyclers"""

import pytest
import jax.numpy as jnp
import jax.random as jrand
import dabench as dab


key = jrand.PRNGKey(42)


@pytest.fixture
def l96_nature_run():
    """Defines class Lorenz96 object for rest of tests."""
    l96 = dab.data.Lorenz96(system_dim=6, store_as_jax=True, delta_t=0.01)
    return l96.generate(n_steps=120)

@pytest.fixture
def obs_vec_l96(l96_nature_run):
    """Generate observations for rest of tests."""
    obs_l96 = dab.observer.Observer(
        l96_nature_run, 
        times=l96_nature_run['time'].data[jnp.arange(0, 120, 5)],
        random_location_count = 3,
        error_bias = 0.0,
        error_sd = 0.3,
        random_seed=94,
        stationary_observers=True,
        store_as_jax=True
    )

    return obs_l96.observe()

@pytest.fixture
def l96_fc_model():
    model_l96 = dab.data.Lorenz96(system_dim=6, store_as_jax=True, delta_t=0.01)

    class L96Model(dab.model.Model):                                                                       
        """Defines model wrapper for Lorenz96 to test forecasting."""
        def forecast(self, state_vec, n_steps):
            new_vec = self.model_obj.generate(x0=state_vec['x'].data, n_steps=n_steps)

            return new_vec.isel(time=-1), new_vec

        def compute_tlm(self, state_vec, n_steps):
            x, M  = self.model_obj.generate(n_steps=n_steps, x0=state_vec['x'].data,
                                            return_tlm=True)
            return x, M

    return L96Model(model_obj=model_l96)

@pytest.fixture
def var4d_cycler(l96_fc_model):
    dc = dab.dacycler.Var4D(
        system_dim=6,
        delta_t=0.01,
        model_obj=l96_fc_model,
        obs_window_indices=[0,5, 10],
        steps_per_window=11
    )
    
    return dc


@pytest.fixture
def var4d_backprop_cycler(l96_fc_model):
    B = jnp.identity(6)*0.05
    dc = dab.dacycler.Var4DBackprop(
        system_dim=6,
        delta_t=0.01,
        model_obj=l96_fc_model,
        obs_window_indices=[0,5, 10],
        steps_per_window=11,
        learning_rate=0.1,
        lr_decay=0.5,
        B=B
    )
    
    return dc

def test_var4d_l96(l96_nature_run, obs_vec_l96, var4d_cycler):
    """Test 4D-Var cycler"""
    init_noise = jrand.normal(key, shape=(6,))
    init_state = l96_nature_run.isel(time=0) + init_noise
    start_time = l96_nature_run['time'].data[0]

    out_sv = var4d_cycler.cycle(
        input_state = init_state,
        start_time = start_time,
        obs_vector = obs_vec_l96,
        obs_error_sd=obs_vec_l96.error_sd*1.5,
        n_cycles=10,  
        analysis_window=0.1,
        return_forecast=True)
    out_sv = out_sv.stack(time=['cycle', 'cycle_timestep']).transpose('time', ...)

    assert out_sv['x'].shape == (100, 6)

    # Check that timeseries is evolving
    assert not jnp.allclose(
        out_sv['x'].values[0,:], 
        out_sv['x'].values[5,:], 
    )
    # Check against presaved results
    assert jnp.allclose(
        out_sv['x'].values[0,:],
        jnp.array([4.27467538,  9.83014683,  2.96253047,  2.88635649, -1.64625228,
                   0.31892547])
    )
    assert jnp.allclose(
        out_sv['x'].values[-1,:],
        jnp.array([-0.06994288,  1.48006508,  6.08807623,  4.65273952,  1.09892658,
                   -4.47113857])
    )

def test_var4d_obs_metrics(l96_nature_run, obs_vec_l96, var4d_cycler):
    """Var4D obs-space metrics: byte-identical analysis, 7 per-cycle scalars,
    spread all-NaN (deterministic, no ensemble), and debug per-obs arrays."""
    import numpy as np
    init_noise = jrand.normal(key, shape=(6,))
    init_state = l96_nature_run.isel(time=0) + init_noise
    kw = dict(input_state=init_state,
              start_time=l96_nature_run['time'].data[0],
              obs_vector=obs_vec_l96, obs_error_sd=obs_vec_l96.error_sd * 1.5,
              n_cycles=10, analysis_window=0.1, return_forecast=True)

    ana_default = var4d_cycler.cycle(**kw)
    ana, metrics = var4d_cycler.cycle(return_metrics=True, **kw)
    assert np.array_equal(np.asarray(ana_default['x'].data),
                          np.asarray(ana['x'].data))

    for var in ("o_minus_f_rms", "o_minus_a_rms", "bias_f", "bias_a",
                "obs_space_spread_background", "obs_space_spread_analysis_end",
                "sigma_obs_max", "n_active_obs"):
        assert metrics[var].shape == (10,)
    n_active = np.asarray(metrics["n_active_obs"].data)
    act = n_active > 0
    assert bool(np.any(act))
    # (5) B-derived spreads: background = obs-space std of the static B (=1 for
    # B=identity); analysis-end = TLM-propagated posterior projected to the end.
    # Both finite/positive on active cycles, and the analysis contracts the
    # covariance (posterior < prior).
    spread_bg = np.asarray(metrics["obs_space_spread_background"].data)[act]
    spread_ana = np.asarray(
            metrics["obs_space_spread_analysis_end"].data)[act]
    assert bool(np.all(np.isfinite(spread_bg))) and bool(np.all(spread_bg > 0))
    assert bool(np.all(np.isfinite(spread_ana))) and bool(
            np.all(spread_ana > 0))
    assert bool(np.all(spread_ana < spread_bg))
    of = np.asarray(metrics["o_minus_f_rms"].data)[act]
    oa = np.asarray(metrics["o_minus_a_rms"].data)[act]
    assert bool(np.all(np.isfinite(of)))
    assert bool(np.all(oa <= of + 1e-6))

    _, metrics_dbg = var4d_cycler.cycle(
        return_metrics=True, metrics_mode="debug", **kw)
    for var in ("o_minus_f", "o_minus_a", "obs_active"):
        assert metrics_dbg[var].dims == ("cycle", "obs")


def test_var4d_backprop_l96(l96_nature_run, obs_vec_l96, var4d_backprop_cycler):
    """Test 4DVar-Backprop cycler"""
    init_noise = jrand.normal(key, shape=(6,))
    init_state = l96_nature_run.isel(time=0) + init_noise
    start_time = l96_nature_run['time'].data[0]

    out_sv = var4d_backprop_cycler.cycle(
        input_state = init_state,
        start_time = start_time,
        obs_vector = obs_vec_l96,
        obs_error_sd=obs_vec_l96.error_sd*1.5,
        n_cycles=10,  
        analysis_window=0.1,
        return_forecast=True)
    out_sv = out_sv.stack(time=['cycle', 'cycle_timestep']).transpose('time', ...)

    assert out_sv['x'].shape == (100, 6)

    # Check that timeseries is evolving
    assert not jnp.allclose(
        out_sv['x'].values[0,:], 
        out_sv['x'].values[5,:], 
    )
    # Check against presaved results
    assert jnp.allclose(
        out_sv['x'].values[0,:], 
        jnp.array([4.66568052,  8.93399413,  3.21968694,  3.12447287, -1.54934608,
                   -0.2022133])
    )
    assert jnp.allclose(
        out_sv['x'].values[-1,:],
        jnp.array([ 1.6213089 ,  3.05965355,  4.37068241,  4.70095984,  4.05523923,
                   -5.03153997])
    )
