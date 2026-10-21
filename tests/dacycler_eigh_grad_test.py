"""Degeneracy-safe gradients of the ETKF/LETKF SPD solve (``_solve_pa_wa``).

``Pa = A^{-1}`` and ``Wa = ((K-1) A^{-1})^{1/2}`` use a Daleckii-Krein
custom JVP, so ``jax.grad`` is finite and exact even for degenerate spectra,
e.g. ``A = (K-1)/rho I`` at a grid point with NO obs in the localisation
radius (plain autodiff of ``eigh`` gives NaN there).  Forward values are
unchanged.
"""
import numpy as np
import pytest
import jax
import jax.numpy as jnp
from dabench.dacycler import LETKF
from dabench.dacycler._utils import _solve_pa_wa

jax.config.update("jax_enable_x64", True)


def _old_pa_wa(A):
    """Pre-fix reference (plain eigh autodiff)."""
    K = A.shape[-1]
    w, V = jnp.linalg.eigh(A)
    inv = jnp.where(w > 0, 1.0 / w, 0.0)
    return (V * inv) @ V.T, (V * jnp.sqrt(jnp.clip((K - 1) * inv, 0.0))) @ V.T


def _spd(rng, K, degenerate=False):
    if degenerate:   # identity-like + rank-1 obs update
        y = rng.standard_normal(K)
        return (K - 1) * np.eye(K) + np.outer(y, y)
    Q, _ = np.linalg.qr(rng.standard_normal((K, K)))
    return (Q * np.linspace(1.0, 5.0, K)) @ Q.T


def _sym(rng, K):
    D = rng.standard_normal((K, K))
    return jnp.asarray(D + D.T)


def _fd(f, x, d, h=1e-6):
    return (f(x + h * d) - f(x - h * d)) / (2 * h)


@pytest.mark.parametrize("impl", [None, "qr"])
def test_forward_identical(impl):
    rng = np.random.default_rng(0)
    for deg in (False, True):
        A = jnp.asarray(_spd(rng, 6, deg))
        Pa, Wa, _ = _solve_pa_wa(A, impl, 40)
        Po, Wo = _old_pa_wa(A)
        np.testing.assert_allclose(Pa, Po, rtol=1e-12, atol=1e-14)
        np.testing.assert_allclose(Wa, Wo, rtol=1e-12, atol=1e-14)


def _loss_of_A(A):
    Pa, Wa, _ = _solve_pa_wa(A, None, 40)
    C = jnp.arange(A.size, dtype=A.dtype).reshape(A.shape) / A.size
    return jnp.sum(C * Pa) + jnp.sum(jnp.sin(C) * Wa ** 2) + jnp.sum(Wa * C.T)


def test_matrix_function_matches_eigh_autodiff_separated():
    rng = np.random.default_rng(1)
    for _ in range(5):
        A = jnp.asarray(_spd(rng, 5))
        g_new = jax.grad(_loss_of_A)(A)

        def old(A):
            Pa, Wa = _old_pa_wa(A)
            C = jnp.arange(A.size, dtype=A.dtype).reshape(A.shape) / A.size
            return (jnp.sum(C * Pa) + jnp.sum(jnp.sin(C) * Wa ** 2)
                    + jnp.sum(Wa * C.T))
        g_old = jax.grad(old)(A)
        # eigh's grad is symmetrised; compare along symmetric directions.
        for _ in range(3):
            d = _sym(rng, 5)
            assert abs(jnp.sum(g_new * d) - jnp.sum(g_old * d)) <= \
                1e-10 * max(1.0, abs(float(jnp.sum(g_old * d))))


@pytest.mark.parametrize("A_kind", ["identity", "rank1", "repeated"])
def test_matrix_function_degenerate_finite_and_fd(A_kind):
    rng = np.random.default_rng(2)
    K = 6
    if A_kind == "identity":
        A = 5.0 * np.eye(K)
    elif A_kind == "rank1":
        A = _spd(rng, K, degenerate=True)
    else:
        Q, _ = np.linalg.qr(rng.standard_normal((K, K)))
        A = (Q * np.array([1., 1., 1., 3., 3., 7.])) @ Q.T
    A = jnp.asarray(A)
    g = jax.grad(_loss_of_A)(A)
    assert np.all(np.isfinite(g))
    for _ in range(4):
        d = _sym(rng, K)
        fd = _fd(_loss_of_A, A, d)
        assert abs(float(jnp.sum(g * d)) - float(fd)) <= 1e-7 * max(1, abs(fd))


# ---------------- LETKF end-to-end ------------------------------------------

def _letkf_case(rng, G=14, n_obs=6, ens=5, mode="gram", **kw):
    # Obs clustered in one cap; grid spread globally, radius 1500 km ->
    # many grid points have ZERO obs in radius.
    obs_ll = np.stack([rng.uniform(10, 30, n_obs), rng.uniform(0, 20, n_obs)], 1)
    grid_ll = np.stack([rng.uniform(-80, 80, G), rng.uniform(0, 360, G)], 1)
    grid_ll[:3] = obs_ll[:3] + 0.5            # some observed points
    c = LETKF(system_dim=G, delta_t=0.01, ensemble_dim=ens, model_obj=None,
              grid_latlon=jnp.asarray(grid_ll), obs_latlon=jnp.asarray(obs_ll),
              localize_radius=1500.0, local_gram=mode, **kw)
    Xb = jnp.asarray(rng.standard_normal((G, ens)))
    Yb = jnp.asarray(rng.standard_normal((n_obs, ens)))
    Y = jnp.asarray(rng.standard_normal(n_obs))
    rinv = jnp.asarray(rng.uniform(0.5, 2.0, n_obs))
    taper = np.asarray(c._build_taper(jnp.arange(n_obs), Xb.dtype))
    unobs = np.all(taper == 0, axis=1)
    assert unobs.sum() >= 3 and (~unobs).sum() >= 3
    return c, Xb, Yb, Y, rinv, jnp.arange(n_obs), unobs


@pytest.mark.parametrize("mode", ["gram", "direct"])
@pytest.mark.parametrize("rtps,rtpp,chunk", [(0., 0., None), (0.5, 0.2, 4)])
def test_letkf_unobserved_points_exact_background(mode, rtps, rtpp, chunk):
    rng = np.random.default_rng(3)
    c, Xb, Yb, Y, rinv, idx, unobs = _letkf_case(
        rng, mode=mode, rtps_relaxation=rtps, rtpp_relaxation=rtpp,
        grid_chunk=chunk)
    Xa = c._localized_analysis(Xb, Yb, Y, rinv, idx, rho=1.0)
    np.testing.assert_array_equal(np.asarray(Xa)[unobs], np.asarray(Xb)[unobs])
    # rho != 1: unobserved points get pure multiplicative inflation.
    rho = 1.21
    Xa = np.asarray(c._localized_analysis(Xb, Yb, Y, rinv, idx, rho=rho))
    xb = np.asarray(Xb)[unobs]
    m = xb.mean(1, keepdims=True)
    pert = np.sqrt(rho) * (xb - m)
    if rtps:   # RTPS relaxes spread back toward sigma_b
        pert = pert * (1 + rtps * (1 / np.sqrt(rho) - 1))
    pert = pert + rtpp * ((xb - m) - pert)
    np.testing.assert_allclose(Xa[unobs], m + pert, rtol=1e-12, atol=1e-13)
    assert not np.allclose(np.asarray(Xa)[~unobs], np.asarray(Xb)[~unobs])


@pytest.mark.parametrize("mode", ["gram", "direct"])
@pytest.mark.parametrize("rtps", [0.0, 0.5])
def test_letkf_grads_finite_and_fd(mode, rtps):
    rng = np.random.default_rng(4)
    c, Xb, Yb, Y, rinv, idx, _ = _letkf_case(
        rng, mode=mode, rtps_relaxation=rtps)

    def loss(Xb, Yb, Y, rinv):
        Xa = c._localized_analysis(Xb, Yb, Y, rinv, idx, rho=1.05)
        m = jnp.mean(Xa, axis=1)
        P = Xa - m[:, None]
        return jnp.sum(m ** 2) + jnp.sum(P ** 2) + jnp.sum(jnp.sin(P))

    args = (Xb, Yb, Y, rinv)
    grads = jax.grad(loss, argnums=(0, 1, 2, 3))(*args)
    for g in grads:
        assert np.all(np.isfinite(g))
    for i, g in enumerate(grads):
        for _ in range(2):
            d = jnp.asarray(rng.standard_normal(args[i].shape))
            f = lambda x: loss(*[x if j == i else a for j, a in enumerate(args)])
            fd = float(_fd(f, args[i], d))
            assert abs(float(jnp.sum(g * d)) - fd) <= 1e-6 * max(1, abs(fd))
