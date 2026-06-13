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
        # Window-start analysis ensemble-mean vs nature (10 model steps/cycle,
        # init at nature index 10).
        ana = np.asarray(out.isel(cycle_timestep=0).mean('ensemble')['x'].data)
        nat = np.asarray(l96_nature_run['x'].data)
        idx = [10 + c * 10 for c in range(ana.shape[0])]
        return float(np.sqrt(np.mean((ana - nat[idx]) ** 2)))

    rmse_da = _rmse(_run(1.0))
    rmse_noop = _rmse(_run(1.0e6))  # huge R -> near no-op update
    assert rmse_da < rmse_noop
