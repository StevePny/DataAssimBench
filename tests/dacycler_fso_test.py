"""Forecast sensitivity to observations (FSO) of the matrix-free incremental 4D-Var, compiled-window path.

For a linear model and observation operator and one outer loop the Gauss-Newton analysis is exact, so
  * the impacts add up to the first-order change of the forecast-error measure, grad(J_e) . (x_a - x_b);
  * the observation-space sensitivity s_i is the derivative of J_e(x_a) with respect to the observation value y_i (finite differences);
  * the impact of an observation without a time/location (masked) is zero.
"""
import jax
import jax.numpy as jnp
import numpy as np
import xarray as xr

import dabench as dab

jax.config.update("jax_enable_x64", True)

N, S, DT = 10, 6, 1.0
LOCS = [0, 2, 4, 6, 8]
A_, B_ = 0.9, 0.08
TRUTH_END = np.sin(np.linspace(0, 3, N))


def step(x, t, u):
    return A_ * x + B_ * jnp.roll(x, 1) + u


class LinModel(dab.model.Model):
    def __init__(self):
        self.model_obj = None
        self.delta_t = DT

    def step_fn(self):
        return step

    def forecast(self, state_vec, n_steps):
        raise NotImplementedError


def make(obs_vals):
    times = np.arange(0, S).astype(float)
    return xr.Dataset(
        {'x': (('time', 'observations'), obs_vals),
         'system_index': (('variable', 'time', 'observations'), np.broadcast_to(np.asarray(LOCS), (1, len(times), len(LOCS))).copy())},
        coords={'time': times, 'observations': np.arange(len(LOCS)), 'variable': ['x']}, attrs=dict(stationary_observers=True, error_sd=0.05))


def cycler():
    return dab.dacycler.Var4DOperator(system_dim=N, delta_t=DT, model_obj=LinModel(), tlm_op_factory=None, B_half_op=lambda v: 0.3 * v, n_outer_loops=1, n_inner_loops=200, inner_tol=1e-13, steps_per_window=S)


def state(x):
    return xr.Dataset({'x': (('index',), np.asarray(x, float))}, coords={'index': np.arange(N), 'time': 0.0})


U = jnp.asarray(0.05 * np.random.default_rng(2).standard_normal((S - 1, N)))


def j_e(x_a):
    x = x_a
    for k in range(S - 1):
        x = step(x, 0.0, U[k])
    return 0.5 * jnp.sum((x - TRUTH_END) ** 2)


def setup(seed=0):
    rng = np.random.default_rng(seed)
    xb = np.cos(np.linspace(0, 2, N)) + 0.2 * rng.standard_normal(N)
    truth = np.cos(np.linspace(0, 2, N))
    traj = [truth]
    for k in range(S - 1):
        traj.append(np.asarray(step(jnp.asarray(traj[-1]), 0.0, U[k])))
    y = np.stack(traj)[:, LOCS] + 0.05 * rng.standard_normal((S, len(LOCS)))
    return xb, y


def run_fso(xb, y, **kw):
    return cycler().fso(state(xb), 0.0, make(y), 0.05, float(S - 1), j_e, model_inputs=U, **kw)


def test_impacts_sum_to_the_first_order_change():
    xb, y = setup()
    r = run_fso(xb, y)
    assert r['impact'].shape == (S, len(LOCS))
    np.testing.assert_allclose(float(r['total']), float(r['total_linear']), rtol=1e-6)
    assert abs(float(r['total'])) > 1e-6  # not vacuous


def test_sensitivity_is_the_derivative_with_respect_to_the_observation():
    xb, y = setup()
    r = run_fso(xb, y)
    s = np.asarray(r['sensitivity'])
    for (k, l) in ((2, 3), (4, 1)):
        eps = 1e-5
        yp, ym = y.copy(), y.copy()
        yp[k, l] += eps
        ym[k, l] -= eps
        fd = (float(j_e(run_fso(xb, yp)['x_a'])) - float(j_e(run_fso(xb, ym)['x_a']))) / (2 * eps)
        np.testing.assert_allclose(s[k, l], fd, rtol=2e-4, atol=1e-9)


def test_zero_weight_observations_have_no_impact_and_per_observation_errors_are_used():
    """An observation whose error is NaN (weight 0 in the per-observation error array) has no impact, and the sensitivities follow the per-observation errors."""
    xb, y = setup()
    sd = np.full((S, len(LOCS)), 0.05)
    sd[3, 2] = np.nan
    sd[1, 1] = 0.5
    r = cycler().fso(state(xb), 0.0, make(y), sd, float(S - 1), j_e, model_inputs=U)
    assert float(r['impact'][3, 2]) == 0.0 and float(r['sensitivity'][3, 2]) == 0.0
    np.testing.assert_allclose(float(r['total']), float(r['total_linear']), rtol=1e-6)
    r0 = run_fso(xb, y)
    assert abs(float(r['sensitivity'][1, 1])) < 0.5 * abs(float(r0['sensitivity'][1, 1]))  # the observation with the larger error carries less weight


# --- weak constraint: controls (v, chi), the forecast carries the fitted model-error forcing ---

def wc_cycler(eta_mode):
    return dab.dacycler.Var4DOperatorWC(system_dim=N, delta_t=DT, model_obj=LinModel(), tlm_op_factory=None, B_half_op=lambda v: 0.3 * v, Q_half=0.05, eta_mode=eta_mode, n_outer_loops=1, n_inner_loops=400, inner_tol=1e-13, steps_per_window=S)


def j_e_wc(x_a, eta):
    x = x_a
    for k in range(S - 1):
        x = step(x, 0.0, U[k]) + (0.0 if eta is None else eta[k])
    return 0.5 * jnp.sum((x - TRUTH_END) ** 2)


def run_wc(xb, y, eta_mode):
    return wc_cycler(eta_mode).fso(state(xb), 0.0, make(y), 0.05, float(S - 1), j_e_wc, model_inputs=U)


import pytest


@pytest.mark.parametrize("eta_mode", ["constant", "per_step"])
def test_weak_constraint_impacts_sum_to_the_first_order_change(eta_mode):
    xb, y = setup()
    r = run_wc(xb, y, eta_mode)
    assert r['impact'].shape == (S, len(LOCS))
    np.testing.assert_allclose(float(r['total']), float(r['total_linear']), rtol=1e-5)
    assert abs(float(r['total'])) > 1e-6


@pytest.mark.parametrize("eta_mode", ["constant", "per_step"])
def test_weak_constraint_sensitivity_is_the_derivative_with_respect_to_the_observation(eta_mode):
    xb, y = setup(1)
    r = run_wc(xb, y, eta_mode)
    s = np.asarray(r['sensitivity'])
    for (k, l) in ((2, 3), (4, 1)):
        eps = 1e-5
        yp, ym = y.copy(), y.copy()
        yp[k, l] += eps
        ym[k, l] -= eps
        a, b = run_wc(xb, yp, eta_mode), run_wc(xb, ym, eta_mode)
        fd = (float(j_e_wc(a['x_a'], a['eta_seq'])) - float(j_e_wc(b['x_a'], b['eta_seq']))) / (2 * eps)
        np.testing.assert_allclose(s[k, l], fd, rtol=5e-4, atol=1e-8)
