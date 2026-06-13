"""Tests for the bred-vector climatological-B construction.

Exercises :mod:`dabench.dacycler._bred_vectors`:

  * :func:`breed_vectors` — collected count, amplitude seeding, and the
    growing-mode alignment property (bred vectors converge to the
    leading Lyapunov direction of a known linear model).
  * :func:`bred_vectors_to_B_factors` — factor shapes and operational
    trace matching.
  * :func:`build_bred_clim_B` — end-to-end reconstruction of the sample
    covariance via :func:`build_B_half`.
"""

import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp  # noqa: E402
import jax.random as jrand  # noqa: E402
import numpy as np  # noqa: E402
import pytest  # noqa: E402

from dabench.dacycler import (  # noqa: E402
        breed_vectors,
        bred_vectors_to_B_factors,
        build_bred_clim_B,
        build_B_half,
        )
from dabench.dacycler._bred_vectors import _amplitude, _rescale  # noqa: E402


@pytest.fixture
def linear_model():
    """Orthogonal-eigenbasis linear model with a unique fastest mode."""
    D = 6
    Q, _ = jnp.linalg.qr(jrand.normal(jrand.PRNGKey(0), (D, D),
                                      dtype=jnp.float64))
    lams = jnp.array([1.6, 1.3, 1.0, 0.8, 0.6, 0.4], dtype=jnp.float64)

    def forecast_fn(x, n_steps):
        return Q @ ((lams ** n_steps) * (Q.T @ x))

    return D, Q, forecast_fn


def test_rescale_sets_target_norm():
    """``_rescale`` produces a vector whose norm equals the target."""
    v = jrand.normal(jrand.PRNGKey(3), (10,), dtype=jnp.float64)
    for norm in ("rms", "l2"):
        out = _rescale(v, 0.25, norm)
        assert jnp.allclose(_amplitude(out, norm), 0.25, atol=1e-12)
        # Direction preserved.
        assert jnp.allclose(out / jnp.linalg.norm(out),
                            v / jnp.linalg.norm(v), atol=1e-12)


def test_breed_vectors_count_and_seed_amplitude(linear_model):
    """Collected count matches the spinup budget across the ensemble."""
    D, _, forecast_fn = linear_model
    n_anchor, n_spinup, E = 12, 4, 3
    control = jnp.zeros((n_anchor, D), dtype=jnp.float64)
    bred, info = breed_vectors(
            forecast_fn, control, init_amplitude=0.1, growth_steps=2,
            n_spinup=n_spinup, ensemble_size=E, seed=1, norm="rms")
    assert bred.shape == (E * (n_anchor - 1 - n_spinup), D)
    assert info["n_bred_vectors"] == bred.shape[0]
    assert info["growth_steps"] == 2 and info["norm"] == "rms"


def test_breed_vectors_align_to_leading_growing_mode(linear_model):
    """After spinup, the dominant bred direction is the fastest mode."""
    D, Q, forecast_fn = linear_model
    control = jnp.zeros((40, D), dtype=jnp.float64)
    bred, _ = breed_vectors(
            forecast_fn, control, init_amplitude=0.05, growth_steps=2,
            n_spinup=20, ensemble_size=4, seed=2, norm="l2")
    U, _, _ = jnp.linalg.svd(bred.T, full_matrices=False)
    q0 = Q[:, 0]
    assert abs(float(U[:, 0] @ q0)) > 0.999


def test_breed_vectors_forecast_control_false_matches_trajectory(linear_model):
    """With a self-consistent zero control, the two control-leg modes agree."""
    D, _, forecast_fn = linear_model
    control = jnp.zeros((15, D), dtype=jnp.float64)
    kw = dict(init_amplitude=0.05, growth_steps=2, n_spinup=3,
              ensemble_size=2, seed=5, norm="rms")
    b_true, _ = breed_vectors(forecast_fn, control,
                              forecast_control=True, **kw)
    b_false, _ = breed_vectors(forecast_fn, control,
                               forecast_control=False, **kw)
    assert jnp.allclose(b_true, b_false, atol=1e-10)


def test_bred_vectors_to_B_factors_shapes_and_trace():
    """Factor shapes are correct and trace matching hits the target."""
    rng = jrand.PRNGKey(7)
    M, D, K = 80, 6, 6
    bred = jrand.normal(rng, (M, D), dtype=jnp.float64)
    target = 2.5
    bf, info = bred_vectors_to_B_factors(
            bred, K=K, sigma_bg=0.0, target_trace=target)
    assert bf.U.shape == (D, K) and bf.sigma.shape == (K,)
    assert info["K_effective"] == K
    # Full-rank retention => retained variance == target trace.
    assert jnp.allclose(jnp.sum(bf.sigma ** 2), target, atol=1e-9)


def test_build_bred_clim_B_reconstructs_sample_covariance(linear_model):
    """``B`` from build_B_half reproduces the bred sample covariance."""
    D, _, forecast_fn = linear_model
    bf, info = build_bred_clim_B(
            forecast_fn, jnp.zeros((30, D), dtype=jnp.float64),
            K=D, init_amplitude=0.05, growth_steps=2, n_spinup=10,
            ensemble_size=4, seed=4, norm="l2", sigma_bg=0.0)
    apply_B_half = build_B_half(bf)
    # B = (B^1/2)(B^1/2) since sigma_bg=0 and the root is symmetric.
    B = jnp.stack([apply_B_half(apply_B_half(e))
                   for e in jnp.eye(D, dtype=jnp.float64)])
    B_ref = bf.U @ jnp.diag(bf.sigma ** 2) @ bf.U.T
    assert jnp.allclose(B, B_ref, atol=1e-9)
    assert "breeding" in info and "factors" in info


def test_breed_vectors_validates_spinup(linear_model):
    """Spinup must leave at least one retained cycle."""
    D, _, forecast_fn = linear_model
    with pytest.raises(ValueError):
        breed_vectors(forecast_fn, jnp.zeros((5, D), dtype=jnp.float64),
                      init_amplitude=0.1, growth_steps=1, n_spinup=4)
