"""Tests for ETKF Data Assimilation Cycler (dabench.dacycler._etkf)"""

import pytest
import numpy as np
import jax.numpy as jnp
import jax.random as jrand
import dabench as dab


key = jrand.PRNGKey(42)


@pytest.fixture
def lorenz96():
    """Defines class Lorenz96 object for rest of tests."""
    l96 = dab.data.Lorenz96(system_dim=5, store_as_jax=True, delta_t=0.01)
    l96.generate(n_steps=25)

    return l96

@pytest.fixture
def obs_vec_l96(lorenz96):
    """Generate observations for rest of tests."""
    obs_l96 = dab.observer.Observer(
        lorenz96,
        time_indices=np.arange(0, 25, 5),
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
    model_l96 = dab.data.Lorenz96(system_dim=5, store_as_jax=True)

    class L96Model(dab.model.Model):                                                                       
        """Defines model wrapper for Lorenz96 to test forecasting."""
        def forecast(self, state_vec):
            # NOTE: n_steps = 2 because the initial state counts as a "step"
            self.model_obj.generate(x0=state_vec.values, n_steps=2)
            new_vals = self.model_obj.values[-1] 

            new_vec = dab.vector.StateVector(values=new_vals, store_as_jax=True)

            return new_vec

    return L96Model(model_obj=model_l96)

@pytest.fixture
def etkf_cycler(l96_fc_model):
    dc = dab.dacycler.ETKF(
        system_dim=5,
        delta_t=0.01,
        ensemble_dim=8,
        model_obj=l96_fc_model)
    
    return dc

def test_etkf_l96(lorenz96, obs_vec_l96, etkf_cycler):

    cur_tstep=10
    init_noise = jrand.normal(key, shape=(8, 5))
    init_state = dab.vector.StateVector(
        values=lorenz96.values[cur_tstep] + init_noise,
        store_as_jax=True)
    start_time = lorenz96.times[cur_tstep]

    out_sv = etkf_cycler.cycle(
        input_state=init_state,
        start_time=start_time,
        obs_vector=obs_vec_l96,
        obs_error_sd=1.5,
        analysis_window=0.1,
        timesteps=10
        )

    out_sv_mean = np.mean(out_sv.values, axis=1)

    assert out_sv.values.shape == (10, 8, 5)
    assert out_sv_mean.shape == (10, 5)
    # Check that ensemble members are different
    assert not jnp.allclose(
        out_sv.values[-1, 1, :],
        out_sv.values[-1, 0, :],
    )
    # Check first cycle against presaved results
    assert jnp.allclose(
        out_sv.values[0, 0, :],
        jnp.array([-2.43407963,  1.92287259,  1.48351731,  6.97069705,  7.22939174])
    )
    # Check last cycle against presaved results
    assert jnp.allclose(
        out_sv.values[-1, 0, :],
        jnp.array([-0.04859429,  4.88513592,  8.61996513,  5.71426263,  1.90274387])
    )
    # Check mean against presaved results
    assert jnp.allclose(
        out_sv_mean[-1, :],
        jnp.array([-0.32955335,  4.76747091,  8.56107589,  5.94954867,  1.97952361])
    )
