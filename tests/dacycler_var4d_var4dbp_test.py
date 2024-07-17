"""Tests for Var4D and Var4D-Backprop Data Assimilation Cyclers"""

import pytest
import jax.numpy as jnp
import jax.random as jrand
import dabench as dab


key = jrand.PRNGKey(42)


@pytest.fixture
def lorenz96():
    """Defines class Lorenz96 object for rest of tests."""
    l96 = dab.data.Lorenz96(system_dim=6, store_as_jax=True, delta_t=0.01)
    l96.generate(n_steps=120)

    return l96

@pytest.fixture
def obs_vec_l96(lorenz96):
    """Generate observations for rest of tests."""
    obs_l96 = dab.observer.Observer(
        lorenz96, 
        time_indices=jnp.arange(0, 120, 5),
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
    model_l96 = dab.data.Lorenz96(system_dim=6, store_as_jax=True)

    class L96Model(dab.model.Model):                                                                       
        """Defines model wrapper for Lorenz96 to test forecasting."""
        def forecast(self, state_vec, n_steps):
            # NOTE: n_steps = 2 because the initial state counts as a "step"
            self.model_obj.generate(x0=state_vec.values, n_steps=n_steps)
            new_vals = self.model_obj.values

            new_vec = dab.vector.StateVector(values=new_vals, store_as_jax=True)

            return new_vec
        def compute_tlm(self, state_vec, n_steps):
            """Compute TLM"""
            M = self.model_obj.generate(n_steps=n_steps, x0=state_vec.values,
                                        return_tlm=True)
            return M, self.model_obj.values

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

def test_var4d_l96(lorenz96, obs_vec_l96, var4d_cycler):
    """Test 4D-Var cycler"""
    init_noise = jrand.normal(key, shape=(6,))
    init_state = dab.vector.StateVector(
        values=lorenz96.values[0] + init_noise,
        store_as_jax=True)
    start_time = lorenz96.times[0]

    out_sv = var4d_cycler.cycle(
        input_state = init_state,
        start_time = start_time,
        obs_vector = obs_vec_l96,
        obs_error_sd=obs_vec_l96.error_sd*1.5,
        timesteps=10,  
        analysis_window=0.1)

    assert out_sv.values.shape == (100, 6)

    # Check that timeseries is evolving
    assert not jnp.allclose(
        out_sv.values[0,:], 
        out_sv.values[5,:], 
    )
    # Check against presaved results
    assert jnp.allclose(
        out_sv.values[0,:], 
        jnp.array([4.5784335 , 10.70937771,  3.97859892,  0.25609285, -1.89681598,
                   -1.34747704])
    )
    assert jnp.allclose(
        out_sv.values[-1,:],
        jnp.array([-2.32350141,  2.66564733,  9.1592932 ,  0.26887161, -2.72295144,
                    1.24513147])
    )

def test_var4d_backprop_l96(lorenz96, obs_vec_l96, var4d_backprop_cycler):
    """Test 4DVar-Backprop cycler"""
    init_noise = jrand.normal(key, shape=(6,))
    init_state = dab.vector.StateVector(
        values=lorenz96.values[0] + init_noise,
        store_as_jax=True)
    start_time = lorenz96.times[0]

    out_sv = var4d_backprop_cycler.cycle(
        input_state = init_state,
        start_time = start_time,
        obs_vector = obs_vec_l96,
        obs_error_sd=obs_vec_l96.error_sd*1.5,
        timesteps=10,  
        analysis_window=0.1)

    assert out_sv.values.shape == (100, 6)

    # Check that timeseries is evolving
    assert not jnp.allclose(
        out_sv.values[0,:], 
        out_sv.values[5,:], 
    )
    # Check against presaved results
    assert jnp.allclose(
        out_sv.values[0,:], 
        jnp.array([4.53548062, 9.00637144, 3.07940726, 3.24252952, -2.77042587,
                   -2.01121753])
    )
    assert jnp.allclose(
        out_sv.values[-1,:],
        jnp.array([3.91514756,  6.5823489 , -1.60393758, -2.85701674, -0.5386405,
                   0.11637277])
    )
