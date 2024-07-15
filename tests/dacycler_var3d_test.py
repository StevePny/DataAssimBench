"""Tests for Var3D Data Assimilation Cycler (dabench.dacycler._var3d)"""

import pytest
import jax.numpy as jnp
import jax.random as jrand
import dabench as dab


key = jrand.PRNGKey(42)


@pytest.fixture
def lorenz96():
    """Defines class Lorenz96 object for rest of tests."""
    l96 = dab.data.Lorenz96(system_dim=6, store_as_jax=True)
    l96.generate(n_steps=50)

    return l96

@pytest.fixture
def obs_vec_l96(lorenz96):
    """Generate observations for rest of tests."""
    obs_l96 = dab.observer.Observer(
        lorenz96, 
        random_time_density = 0.7,
        random_location_count = 3,
        error_bias = 0.0,
        error_sd = 0.7,
        random_seed=94,
        stationary_observers=True
    )

    return obs_l96.observe()

@pytest.fixture
def l96_fc_model():
    model_l96 = dab.data.Lorenz96(system_dim=6)

    class L96Model(dab.model.Model):                                                                       
        """Defines model wrapper for Lorenz96 to test forecasting."""
        def forecast(self, state_vec):
            # NOTE: n_steps = 2 because the initial state counts as a "step"
            self.model_obj.generate(x0=state_vec.values[:,0], n_steps=2)
            new_vals = self.model_obj.values[-1] 

            new_vec = dab.vector.StateVector(values=new_vals, store_as_jax=True)

            return new_vec

    return L96Model(model_obj=model_l96)

@pytest.fixture
def var3d_cycler(l96_fc_model):
    dc = dab.dacycler.Var3D(
        system_dim=6,
        delta_t=0.05,
        model_obj=l96_fc_model)
    
    return dc

def test_var3d_l96(lorenz96, obs_vec_l96, var3d_cycler):

    # Adding some noise to our initial state and getting the start time in model units
    init_noise = jrand.normal(key, shape=(6,))
    init_state = dab.vector.StateVector(
        values=lorenz96.values[0] + init_noise,
        store_as_jax=True)
    start_time = lorenz96.times[0]

    # To run the experiment, we use the cycle() method:
    out_sv = var3d_cycler.cycle(
        input_state = init_state,
        start_time = start_time,
        obs_vector = obs_vec_l96,
        timesteps=10, 
        analysis_window=0.25)

    assert out_sv.values.shape == (10, 6, 1)
    assert jnp.allclose(
        out_sv.values[0,:,0], 
        # Presaved results
        jnp.array([-0.90632236, 1.20601681, 1.64865068, 5.03383547, 0.60286713, -3.75779771])
    )
    assert jnp.allclose(
        out_sv.values[-1,:,0],
        jnp.array([2.4359271 , 6.33357301, 3.1125237, -3.13591255, 0.6081794, 1.51721364])
    )