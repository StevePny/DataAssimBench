"""Var4DOperator with a nonlinear observation operator ``h``.

Every outer loop recomputes the innovations y - h(x) on its trajectory and
relinearises H' = jvp(h) (compiled-window path). Checks:

  * a linear ``h`` (x -> H x) gives the matrix-H analysis to roundoff;
  * a nonlinear ``h``: the analysis lowers the nonlinear cost J(x) of the
    window, and more outer loops lower it further.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from dabench.dacycler import Var4DOperator
from dabench.dacycler._var4d_operator_utils import BFactors

from dacycler_per_time_h_test import (D, DT, N, OBS_SD, L96RK4Model,  # noqa: F401
                                      _pool_obs, nature, tlm_factory)

jax.config.update("jax_enable_x64", True)

SIGMA_BG = float(np.sqrt(0.5))
OWI = [0, 5, 10]
STEPS = 11


class _L96Step(L96RK4Model):
    def step_fn(self):
        return self.step


def _cycler(H=None, h=None, n_outer=2):
    return Var4DOperator(
        system_dim=D, delta_t=DT, model_obj=_L96Step(), H=H, h=h,
        obs_window_indices=OWI, steps_per_window=STEPS,
        tlm_op_factory=tlm_factory, n_outer_loops=n_outer, n_inner_loops=40,
        B_factors=BFactors(U=jnp.zeros((D, 0)), sigma=jnp.zeros((0,)),
                           sigma_bg=SIGMA_BG))


def _analysis(dc, nature, obs):
    init = nature.isel(time=0) + 0.3 * jax.random.normal(jax.random.PRNGKey(0), (D,))
    out = dc.cycle(input_state=init, start_time=nature['time'].data[0],
                   obs_vector=obs, obs_error_sd=OBS_SD, n_cycles=1,
                   analysis_window=0.1, compile_once=True)
    return init['x'].data, np.asarray(out['x'].data).reshape(-1)


def _H():
    return jax.random.normal(jax.random.PRNGKey(3), (N, D), dtype=jnp.float64)


def _h_nonlinear(H):
    return lambda x: H @ x + 0.05 * (H @ x) ** 2


def test_linear_h_equals_matrix_path(nature):
    H = _H()
    obs = _pool_obs(nature, H, stationary=True)
    _, xa_mat = _analysis(_cycler(H=H), nature, obs)
    _, xa_fun = _analysis(_cycler(h=lambda x: H @ x), nature, obs)
    assert np.max(np.abs(xa_fun - xa_mat)) < 1e-10 * np.max(np.abs(xa_mat))


def test_nonlinear_h_lowers_nonlinear_cost(nature):
    H = _H()
    h = _h_nonlinear(H)
    times = np.asarray(nature['time'].data)
    y_idx = np.arange(0, 80, 5)[:len(OWI)]
    x_true = np.asarray(nature['x'].data)
    rng = np.random.default_rng(7)
    obs = _pool_obs(nature, H, stationary=True)
    y = np.stack([np.asarray(h(jnp.asarray(x_true[i]))) for i in np.arange(0, 80, 5)])
    obs['x'] = (obs['x'].dims, jnp.asarray(y + OBS_SD * rng.standard_normal(y.shape)))
    step = L96RK4Model().step

    def cost(x0, xb):
        xs = [x0]
        for _ in range(STEPS - 1):
            xs.append(step(xs[-1]))
        jo = sum(jnp.sum((obs['x'].data[k] - h(xs[i])) ** 2)
                 for k, i in enumerate(OWI)) / OBS_SD ** 2
        return 0.5 * float(jnp.sum((x0 - xb) ** 2) / SIGMA_BG ** 2 + jo)

    xb, xa1 = _analysis(_cycler(h=h, n_outer=1), nature, obs)
    _, xa3 = _analysis(_cycler(h=h, n_outer=3), nature, obs)
    assert np.all(np.isfinite(xa3))
    j_b, j_1, j_3 = cost(xb, xb), cost(xa1, xb), cost(xa3, xb)
    print(f"nonlinear J: background {j_b:.4e}, 1 outer {j_1:.4e}, 3 outer {j_3:.4e}")
    assert j_1 < j_b and j_3 <= j_1 * (1 + 1e-6)
    del times, y_idx
