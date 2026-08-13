"""Tests for Var3D Data Assimilation Cycler (dabench.dacycler._var3d)"""

import pytest
import numpy as np
import jax
import jax.numpy as jnp
import jax.random as jrand
import dabench as dab

jax.config.update("jax_enable_x64", True)

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


# ── 3D-Var-FGAT: multi-obs-time windows ─────────────────────────────────────
@pytest.fixture
def fgat_nature():
    l96 = dab.data.Lorenz96(system_dim=5, store_as_jax=True, delta_t=0.01)
    return l96.generate(n_steps=200)


@pytest.fixture
def fgat_fc_model():
    m = dab.data.Lorenz96(system_dim=5, store_as_jax=True, delta_t=0.01)

    class L96Model(dab.model.Model):
        def forecast(self, state_vec, n_steps):
            v = self.model_obj.generate(x0=state_vec['x'].data, n_steps=n_steps)
            return v.isel(time=-1).assign_attrs(delta_t=0.01), v

    return L96Model(model_obj=m)


def _dense_obs(nature, sd):
    return dab.observer.Observer(
        nature, times=nature['time'].data[np.arange(0, 200, 5)],
        random_location_count=5, error_bias=0.0, error_sd=sd,
        random_seed=91, stationary_observers=True, store_as_jax=True).observe()


def _cycle_ic_rmse(cyc, nature, obs, sd):
    init = nature.isel(time=10)
    out = cyc.cycle(input_state=init, start_time=init['time'].data,
                    obs_vector=obs, obs_error_sd=sd, analysis_window=0.10,
                    n_cycles=15, return_forecast=True)
    ic = np.asarray(out.isel(cycle_timestep=0)['x'].data)
    nat = np.asarray(nature['x'].data)
    idx = [10 + c * 10 for c in range(ic.shape[0])]
    return float(np.sqrt(np.mean((ic - nat[idx]) ** 2)))


def test_var3d_fgat_beats_legacy_multitime(fgat_nature, fgat_fc_model):
    """With an analysis window spanning MULTIPLE obs times, plain 3D-Var
    collapses all in-window obs onto one state and cannot fit them; 3D-Var-FGAT
    forms the innovation at each obs's true time and drives the error far lower
    (dense, near-perfect obs)."""
    obs = _dense_obs(fgat_nature, 1e-3)
    legacy = dab.dacycler.Var3D(system_dim=5, delta_t=0.01,
                                model_obj=fgat_fc_model)
    fgat = dab.dacycler.Var3D(system_dim=5, delta_t=0.01,
                              model_obj=fgat_fc_model, fgat=True)
    rmse_legacy = _cycle_ic_rmse(legacy, fgat_nature, obs, 1e-3)
    rmse_fgat = _cycle_ic_rmse(fgat, fgat_nature, obs, 1e-3)
    # FGAT drives the dense-perfect-obs error to near zero; legacy is stuck.
    assert rmse_fgat < 0.05
    assert rmse_fgat < 0.25 * rmse_legacy


@pytest.mark.parametrize("ati", ["start", "mid", "end"])
def test_var3d_fgat_analysis_time_placement(fgat_nature, fgat_fc_model, ati):
    """3D-Var-FGAT must produce a finite, error-reducing analysis for the
    analysis time placed at the window START, MIDDLE, and END.  The IC handoff
    always re-forecasts tau -> window end, so ``cycle_timestep=0`` is the
    window-end analysis for every placement."""
    obs = _dense_obs(fgat_nature, 1e-3)
    fgat = dab.dacycler.Var3D(system_dim=5, delta_t=0.01,
                              model_obj=fgat_fc_model, fgat=True,
                              analysis_time_index=ati)
    rmse_fgat = _cycle_ic_rmse(fgat, fgat_nature, obs, 1e-3)
    # Dense, near-perfect obs -> every placement drives the error near zero.
    assert np.isfinite(rmse_fgat)
    assert rmse_fgat < 0.05


def test_var3d_fgat_bad_analysis_time_index_raises(fgat_fc_model):
    with pytest.raises(ValueError, match="analysis_time_index"):
        c = dab.dacycler.Var3D(system_dim=5, delta_t=0.01,
                               model_obj=fgat_fc_model, fgat=True,
                               analysis_time_index="bogus")
        c.steps_per_window = 11
        c._resolve_analysis_index()
