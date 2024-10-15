"""Tests for Var3D Data Assimilation Cycler (dabench.dacycler._var3d)"""

import pytest
import jax.numpy as jnp
import jax.random as jrand
import dabench as dab


key = jrand.PRNGKey(42)


@pytest.fixture
def l96_nature_run():
    """Defines class Lorenz96 object for rest of tests."""
    l96 = dab.data.Lorenz96(system_dim=6, store_as_jax=True)
    traj = l96.generate(n_steps=50)

    return traj

@pytest.fixture
def obs_vec_l96(l96_nature_run):
    """Generate observations for rest of tests."""
    obs_l96 = dab.observer.Observer(
        l96_nature_run, 
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
    model_l96 = dab.data.Lorenz96(system_dim=6, store_as_jax=True)

    class L96Model(dab.model.Model):                                                                       
        """Defines model wrapper for Lorenz96 to test forecasting."""
        def forecast(self, state_vec, n_steps):
            new_vec = self.model_obj.generate(x0=state_vec['x'].data, n_steps=n_steps)

            return new_vec.isel(time=-1), new_vec

    return L96Model(model_obj=model_l96)

@pytest.fixture
def var3d_cycler(l96_fc_model):
    dc = dab.dacycler.Var3D(
        system_dim=6,
        delta_t=0.05,
        model_obj=l96_fc_model)
    
    return dc

def test_var3d_l96(l96_nature_run, obs_vec_l96, var3d_cycler):

    # Adding some noise to our initial state and getting the start time in model units
    init_noise = jrand.normal(key, shape=(6,))
    init_state = l96_nature_run.isel(time=0) + init_noise
    start_time = l96_nature_run['time'].values[0]

    # To run the experiment, we use the cycle() method:
    out_sv = var3d_cycler.cycle(
        input_state = init_state,
        start_time = start_time,
        obs_vector = obs_vec_l96,
        n_cycles=10,
        analysis_window=0.25,
        return_forecast=False)

    assert out_sv['x'].shape == (10,6)
    assert jnp.allclose(
        out_sv['x'].values[0],
        # Presaved results
        jnp.array([-0.90632236, -1.20861455, 1.64865068,
                   5.11034063, 4.399881, -3.75779771])
    )
    assert jnp.allclose(
        out_sv['x'].values[-1],
        jnp.array([3.92060079, 3.97290102, -0.763032,
                   -1.5979558, -0.0086728, 2.60395146])
    )
