"""Tests for ETKF Data Assimilation Cycler (dabench.dacycler._etkf)"""

import pytest
import numpy as np
import jax.numpy as jnp
import jax.random as jrand
import dabench as dab


key = jrand.PRNGKey(42)


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
