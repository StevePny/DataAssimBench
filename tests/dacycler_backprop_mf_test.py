"""Matrix-free backpropagation 4D-Var (Var4DBackpropMF) on the compiled-window path.

  * the Lanczos Ritz pairs and the limited-memory inverse are exact for an
    explicit symmetric matrix when the Lanczos steps span the space;
  * on the toy window the analysis reaches the minimum the incremental
    Var4DOperator reaches (same cost and data), and moves towards the truth;
  * the Gauss-Newton preconditioner speeds up the convergence of the gradient
    steps (same learning rate and number of steps, with and without it).
"""
import os
import sys

import jax
import jax.numpy as jnp
import numpy as np

import dabench as dab
from dabench.dacycler._var4d_backprop_mf import lanczos_eigs, lm_inverse

sys.path.insert(0, os.path.dirname(__file__))
from dacycler_compiled_window_test import (DT, LOCS, N, N_CYCLES, S, SIG_B, InputsModel,  # noqa: E402
                                           _setup, obs_vector, state)

jax.config.update("jax_enable_x64", True)


def test_lanczos_and_inverse_on_a_matrix():
    rng = np.random.default_rng(0)
    M = rng.standard_normal((8, 8))
    A = jnp.asarray(np.eye(8) + M @ M.T)
    lam, U = lanczos_eigs(lambda p: A @ p, jnp.asarray(rng.standard_normal(8)), 8)
    np.testing.assert_allclose(np.sort(np.asarray(lam)), np.linalg.eigvalsh(np.asarray(A)), rtol=1e-8)
    P = lm_inverse(lam, U)
    g = jnp.asarray(rng.standard_normal(8))
    np.testing.assert_allclose(np.asarray(P(A @ g)), np.asarray(g), rtol=1e-8, atol=1e-10)


def _run(cls, **kw):
    f, truth, xb = _setup()
    obs = obs_vector(truth, LOCS)
    dc = cls(system_dim=N, delta_t=DT, model_obj=InputsModel(), tlm_op_factory=None,
             B_half_op=lambda v: SIG_B * v, steps_per_window=S, **kw)
    out = dc.cycle(input_state=state(xb), start_time=0.0, obs_vector=obs, obs_error_sd=0.05, n_cycles=N_CYCLES,
                   analysis_window=float(S - 1), return_forecast=True, model_inputs=jnp.asarray(f))
    return np.asarray(out['x'].data), truth, xb


def test_reaches_the_incremental_minimum_and_the_truth():
    a_op, truth, xb = _run(dab.dacycler.Var4DOperator, n_outer_loops=8, n_inner_loops=100, inner_tol=1e-14)
    a_bp, _, _ = _run(dab.dacycler.Var4DBackpropMF, num_iters=200, learning_rate=1.0, lr_decay=0.95, n_lanczos=N)
    err_b = np.abs(xb - truth[0]).mean()
    assert np.abs(a_bp[0, 0] - truth[0]).mean() < err_b
    # first window (same background, cost and data): the two minimisers agree (a constant rate of 1 would oscillate 0.065 away)
    assert np.abs(a_bp[0, 0] - a_op[0, 0]).max() < 1e-4
    # later, more nonlinear windows: the incremental analysis approaches the backprop one only with more outer loops (0.11, 0.09, 0.07 apart
    # with 4, 8, 16), and the backprop analysis is at least as close to the truth
    err = lambda a, c: np.abs(a[c, 0] - truth[c * (S - 1)]).mean()  # noqa: E731
    assert all(err(a_bp, c) <= err(a_op, c) + 1e-6 for c in range(N_CYCLES))


def test_preconditioner_speeds_up_convergence():
    a_op, truth, _ = _run(dab.dacycler.Var4DOperator, n_outer_loops=4, n_inner_loops=60, inner_tol=1e-14)
    a_p, _, _ = _run(dab.dacycler.Var4DBackpropMF, num_iters=3, learning_rate=1.0, lr_decay=1.0, n_lanczos=N)
    a_0, _, _ = _run(dab.dacycler.Var4DBackpropMF, num_iters=3, learning_rate=0.1, lr_decay=1.0, n_lanczos=0)
    d = lambda a: np.abs(a[0, 0] - a_op[0, 0]).mean()  # noqa: E731
    assert d(a_p) < 0.5 * d(a_0)


def test_weak_constraint_reaches_the_incremental_weak_minimum():
    """Var4DBackpropMFWC (controls: initial state and a constant model error) against Var4DOperatorWC on the same window, Q and data."""
    q = 0.02
    a_op, truth, xb = _run(dab.dacycler.Var4DOperatorWC, Q_half=q, eta_mode='constant', n_outer_loops=8, n_inner_loops=200, inner_tol=1e-14)
    a_bp, _, _ = _run(dab.dacycler.Var4DBackpropMFWC, Q_half=q, eta_mode='constant', num_iters=300, learning_rate=1.0, lr_decay=0.97, n_lanczos=2 * N)
    assert np.all(np.isfinite(a_bp))
    assert np.abs(a_bp[0, 0] - truth[0]).mean() < np.abs(xb - truth[0]).mean()
    assert np.abs(a_bp[0, 0] - a_op[0, 0]).max() < 1e-3


def test_weak_constraint_without_q_is_the_strong_backprop():
    a_wc, _, _ = _run(dab.dacycler.Var4DBackpropMFWC, num_iters=50, learning_rate=1.0, lr_decay=0.95, n_lanczos=N)
    a_sc, _, _ = _run(dab.dacycler.Var4DBackpropMF, num_iters=50, learning_rate=1.0, lr_decay=0.95, n_lanczos=N)
    np.testing.assert_allclose(a_wc, a_sc, rtol=1e-10, atol=1e-12)


def test_backtrack_refuses_bad_steps():
    from dabench.dacycler._var4d_backprop_mf import backtrack
    cost = lambda x: jnp.where(jnp.abs(x[0]) > 1.0, jnp.nan, jnp.sum(x ** 2))  # noqa: E731
    x = jnp.asarray([0.5, 0.0])
    J, g = jax.value_and_grad(cost)(x)
    # a step to 4.5 is NaN, to 2.5 NaN, to 1.5 NaN, to 1.0 (t = 1/8) is 1.0 > 0.25: worse; at most 2 halvings -> no move
    np.testing.assert_allclose(np.asarray(backtrack(cost, x, jnp.asarray([4.0, 0.0]), J, g, 2)), np.asarray(x))
    # a step towards 0 is found after a few halvings and decreases the cost
    x2 = backtrack(cost, x, jnp.asarray([-4.0, 0.0]), J, g, 6)
    assert float(cost(x2)) < float(J)


def test_line_search_converges_with_a_constant_rate():
    a_op, truth, xb = _run(dab.dacycler.Var4DOperator, n_outer_loops=8, n_inner_loops=100, inner_tol=1e-14)
    a_ls, _, _ = _run(dab.dacycler.Var4DBackpropMF, num_iters=200, learning_rate=1.0, lr_decay=1.0, n_lanczos=N, line_search=8)
    assert np.all(np.isfinite(a_ls))
    assert np.abs(a_ls[0, 0] - a_op[0, 0]).max() < 1e-3  # a constant rate of 1 alone stays 0.065 away (oscillates)
    a_wc, _, _ = _run(dab.dacycler.Var4DBackpropMFWC, Q_half=0.02, eta_mode='constant', num_iters=300, learning_rate=1.0, lr_decay=1.0, n_lanczos=2 * N, line_search=8)
    a_iwc, _, _ = _run(dab.dacycler.Var4DOperatorWC, Q_half=0.02, eta_mode='constant', n_outer_loops=8, n_inner_loops=200, inner_tol=1e-14)
    assert np.abs(a_wc[0, 0] - a_iwc[0, 0]).max() < 1e-3
