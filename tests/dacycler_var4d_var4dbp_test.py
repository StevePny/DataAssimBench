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
