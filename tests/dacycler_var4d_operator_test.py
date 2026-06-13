"""Tests for matrix-free 4D-Var primitives (Var4DOperator utilities).

Exercises the physics-agnostic building blocks in
:mod:`dabench.dacycler._var4d_operator_utils`:

  * :func:`pcg_lanczos_solve` — solves SPD systems to tolerance.
  * :func:`build_B_half` — recovers ``B^(1/2)`` from BFactors.
  * :func:`extract_B_factors` — randomised SVD of a matrix-free op.
  * :func:`window_tlm_rollout` — matches explicit matrix product.
  * :func:`quadratic_cost` — gradient vanishes at the analytic
    optimum of a small linear-Gaussian setup.

Full ``Var4DOperator.cycle`` end-to-end coverage with a real
forecast model is exercised by the MLTLM OSSE smoke test; here we
restrict to closed-form unit verification.
"""

import jax
import jax.numpy as jnp
import jax.random as jrand
import numpy as np
import pytest

from dabench.dacycler._var4d_operator_utils import (
        BFactors,
        build_B_half,
        extract_B_factors,
        pcg_lanczos_solve,
        quadratic_cost,
        window_tlm_rollout,
        )


@pytest.fixture
def spd_system():
    """Small SPD ``A`` and rhs ``b`` for PCG verification."""
    rng = jrand.PRNGKey(0)
    n = 12
    G = jrand.normal(rng, (n, n), dtype=jnp.float64)
    A = G @ G.T + 0.5 * jnp.eye(n, dtype=jnp.float64)
    b = jrand.normal(jrand.PRNGKey(1), (n,), dtype=jnp.float64)
    return A, b


def test_pcg_lanczos_solve_synthetic_spd(spd_system):
    """PCG drives the residual below the requested tolerance."""
    A, b = spd_system
    tol = 1e-8
    x, info = pcg_lanczos_solve(
            lambda v: A @ v, b, max_iter=200, tol=tol)
    r = b - A @ x
    assert info["converged"]
    assert float(jnp.linalg.norm(r)) <= tol * float(jnp.linalg.norm(b)) * 10
    x_ref = jnp.linalg.solve(A, b)
    assert jnp.allclose(x, x_ref, atol=1e-6)


def test_pcg_lanczos_solve_zero_rhs(spd_system):
    """Zero rhs returns zero solution without iterating."""
    A, _ = spd_system
    n = A.shape[0]
    x, info = pcg_lanczos_solve(
            lambda v: A @ v, jnp.zeros(n, dtype=jnp.float64),
            max_iter=50, tol=1e-10)
    assert jnp.allclose(x, jnp.zeros(n))
    assert int(info["n_iter"]) <= 1


def test_build_B_half_identity_when_sigma_bg_only():
    """Empty retained subspace + sigma_bg=alpha gives B^(1/2) = alpha I."""
    D = 8
    alpha = 0.3
    bf = BFactors(
            U=jnp.zeros((D, 0), dtype=jnp.float64),
            sigma=jnp.zeros((0,), dtype=jnp.float64),
            sigma_bg=alpha,
            )
    apply_B_half = build_B_half(bf)
    v = jrand.normal(jrand.PRNGKey(7), (D,), dtype=jnp.float64)
    assert jnp.allclose(apply_B_half(v), alpha * v, atol=1e-12)


def test_build_B_half_known_factors():
    """``B^(1/2) v`` matches its closed-form on a known factorisation."""
    D, K = 10, 3
    rng = jrand.PRNGKey(11)
    G = jrand.normal(rng, (D, K), dtype=jnp.float64)
    U, _ = jnp.linalg.qr(G)
    sigma = jnp.array([2.0, 1.0, 0.5], dtype=jnp.float64)
    sigma_bg = 0.1
    bf = BFactors(U=U, sigma=sigma, sigma_bg=sigma_bg)
    apply_B_half = build_B_half(bf)
    v = jrand.normal(jrand.PRNGKey(13), (D,), dtype=jnp.float64)
    UTv = U.T @ v
    expected = U @ ((sigma - sigma_bg) * UTv) + sigma_bg * v
    assert jnp.allclose(apply_B_half(v), expected, atol=1e-12)


def test_extract_B_factors_low_rank_recovers_subspace():
    """RSVD on a rank-K operator recovers the column space of ``M``."""
    D, K, E = 20, 4, 64
    rng = jrand.PRNGKey(17)
    M_full = jrand.normal(rng, (D, K), dtype=jnp.float64)
    Q_truth, _ = jnp.linalg.qr(M_full)

    def apply_M(z):
        return M_full @ (M_full.T @ z)

    bf = extract_B_factors(apply_M, system_dim=D, K=K, E=E, seed=23)
    assert bf.U.shape == (D, K)
    assert bf.sigma.shape == (K,)
    # Projection onto the recovered subspace should equal projection
    # onto the truth subspace (up to sign / rotation within the span).
    P_recov = bf.U @ bf.U.T
    P_truth = Q_truth @ Q_truth.T
    assert jnp.allclose(P_recov, P_truth, atol=5e-4)


def test_window_tlm_rollout_matches_explicit_matmul():
    """Matrix-free rollout reproduces ``M_T ... M_1 dx_0`` for a
    state-dependent linear TLM."""
    D, T = 6, 5
    rng = jrand.PRNGKey(29)
    keys = jrand.split(rng, T + 2)
    x_traj = jnp.stack([jrand.normal(k, (D,), dtype=jnp.float64)
                        for k in keys[:T + 1]])
    dx0 = jrand.normal(keys[-1], (D,), dtype=jnp.float64)

    # State-dependent diagonal TLM: M(x_t) = diag(1 + 0.1 * x_t).
    def tlm_op(x_t, dx_t):
        return (1.0 + 0.1 * x_t) * dx_t

    dx_traj = window_tlm_rollout(tlm_op, x_traj, dx0)

    dx_ref = dx0
    refs = [dx_ref]
    for t in range(T):
        dx_ref = (1.0 + 0.1 * x_traj[t]) * dx_ref
        refs.append(dx_ref)
    expected = jnp.stack(refs)
    assert dx_traj.shape == (T + 1, D)
    assert jnp.allclose(dx_traj, expected, atol=1e-12)


def test_quadratic_cost_zero_gradient_at_known_optimum():
    """At ``dx0 = analytic increment``, ``grad J(delta_v=0)`` vanishes."""
    D = 5
    H = jnp.eye(D, dtype=jnp.float64)
    R_inv_diag = jnp.ones(D, dtype=jnp.float64)
    x_b_traj = jnp.zeros((2, D), dtype=jnp.float64)
    innovations = jrand.normal(jrand.PRNGKey(41), (1, D), dtype=jnp.float64)
    obs_window_indices = jnp.array([0], dtype=jnp.int32)
    obs_time_mask = jnp.array([True])

    apply_B_half = lambda v: v  # B = I  # noqa: E731

    def tlm_op(x_t, dx_t):
        del x_t
        return dx_t

    # With B=I, H=I, R^-1=I, x_b=0, T=0: optimum is
    # v_total = 0.5 * innovations (standard 3D-Var ridge).
    v_opt = 0.5 * innovations[0]

    def J(dv):
        return quadratic_cost(
                dv, v_opt, tlm_op=tlm_op, x_b_traj=x_b_traj,
                Hs=H[None, :, :], innovations=innovations,
                obs_window_indices=obs_window_indices,
                obs_time_mask=obs_time_mask, R_inv_diag=R_inv_diag,
                apply_B_half=apply_B_half)

    g = jax.grad(J)(jnp.zeros(D, dtype=jnp.float64))
    assert jnp.allclose(g, jnp.zeros(D), atol=1e-10)
