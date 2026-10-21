"""Matrix-free (function) observation operator.

:class:`LinearObsOperator` wraps a linear JAX function ``x -> y`` (here a
gather from a few derived "component fields" of the state, the structure of
a real network operator) so the cyclers never hold an ``(N, D)`` H.  Checks:

  * ``H @ x``, ``H @ X`` and ``H.T @ r`` equal the dense matrix products;
    adjoint identity ``<Hx, y> == <x, H^T y>``;
  * Var4DOperator / LETKF4D analyses with the function operator equal the
    matrix path to roundoff (stationary and NaN-masked pools);
  * the traced Var4DOperator window has no ``(N, D)``-shaped array.
"""

import jax
import jax.numpy as jnp
import jax.random as jrand
import numpy as np
import pytest

from dabench.dacycler import LETKF4D, LinearObsOperator, Var4DOperator
from dabench.dacycler._per_time_h import PerTimeH

from dacycler_per_time_h_test import (D, N, OBS_SD, _pool_obs, _run_letkf4d,
                                      _run_var, _shapes_in_jaxpr, _var_cycler,
                                      nature)  # noqa: F401

jax.config.update("jax_enable_x64", True)

_IDX = np.random.default_rng(5).choice(3 * D, N, replace=False)


def _fields(x):
    """Three linear 'component fields' of the state, concatenated (3D,)."""
    return jnp.concatenate([x, 0.7 * jnp.cumsum(x),
                            jnp.roll(x, 1) - 0.3 * x])


def _fn(x):
    return _fields(x)[_IDX]


@pytest.fixture(scope="module")
def ops():
    op = LinearObsOperator(_fn, (N, D), jnp.float64)
    return op, np.asarray(op.dense())


def test_function_operator_products_and_adjoint(ops):
    op, H = ops
    x = jrand.normal(jrand.PRNGKey(2), (D,))
    X = jrand.normal(jrand.PRNGKey(3), (D, 5))
    r = jrand.normal(jrand.PRNGKey(4), (N,))
    R = jrand.normal(jrand.PRNGKey(5), (N, 3))
    np.testing.assert_allclose(op @ x, H @ x, atol=1e-14)
    np.testing.assert_allclose(op @ X, H @ X, atol=1e-14)
    np.testing.assert_allclose(op.T @ r, H.T @ r, atol=1e-14)
    np.testing.assert_allclose(op.T @ R, H.T @ R, atol=1e-14)
    lhs, rhs = jnp.dot(op @ x, r), jnp.dot(x, op.T @ r)
    assert abs(float(lhs - rhs)) < 1e-13 * abs(float(lhs))
    # dtype follows a matrix of op.dtype
    assert (op @ x.astype(jnp.float32)).dtype == jnp.float64
    assert (op.astype(jnp.float32) @ x.astype(jnp.float32)).dtype \
        == jnp.float32
    # PerTimeH wrapping: mask on the product, grads = masked adjoint
    T = 3
    mask = jrand.bernoulli(jrand.PRNGKey(1), 0.5, (T, N))
    lazy = PerTimeH(op, T, mask)
    dense = mask[:, :, None] * H
    np.testing.assert_allclose(lazy.dense(), dense, atol=1e-14)
    for t in range(T):
        np.testing.assert_allclose(lazy[t] @ x, dense[t] @ x, atol=1e-14)
        g = jax.grad(lambda z: jnp.dot(r, lazy[t] @ z))(x)
        np.testing.assert_allclose(g, dense[t].T @ r, atol=1e-14)
    # pytree with no leaves (passes through jit as a constant)
    assert jax.tree_util.tree_leaves(op) == []
    np.testing.assert_allclose(jax.jit(lambda h, z: h @ z)(op, x), H @ x,
                               atol=1e-14)


@pytest.mark.parametrize("stationary", [True, False])
def test_var4d_operator_function_equals_matrix(nature, ops, stationary):
    op, H = ops
    obs = _pool_obs(nature, H, stationary)
    mat = _run_var(Var4DOperator, nature, obs, jnp.asarray(H))
    fun = _run_var(Var4DOperator, nature, obs, op)
    assert np.all(np.isfinite(fun))
    d = np.max(np.abs(fun - mat)) / np.max(np.abs(mat))
    print(f"Var4DOperator function vs matrix stationary={stationary} "
          f"max rel diff={d:.3e}")
    assert d < 1e-10


@pytest.mark.parametrize("stationary", [True, False])
def test_letkf4d_function_equals_matrix(nature, ops, stationary):
    op, H = ops
    obs = _pool_obs(nature, H, stationary)
    mat = _run_letkf4d(nature, obs, jnp.asarray(H))
    fun = _run_letkf4d(nature, obs, op)
    assert np.all(np.isfinite(fun))
    d = np.max(np.abs(fun - mat)) / np.max(np.abs(mat))
    print(f"LETKF4D function vs matrix stationary={stationary} "
          f"max rel diff={d:.3e}")
    assert d < 1e-10


def test_function_path_has_no_ND_array(nature, ops):
    op, H = ops
    obs = _pool_obs(nature, H, stationary=False)
    T = 3
    x0 = nature.isel(time=0)
    mask = jnp.asarray(~np.isnan(np.asarray(obs['x'].data[:T])))

    def traced(Hop):
        dc = _var_cycler(Var4DOperator, Hop)
        dc._obs_vector = obs.fillna(0)
        dc.obs_error_sd = OBS_SD
        dc._return_metrics = False

        def f(xb, yv, m):
            return dc._cycle_obsop(
                    x0.assign(x=(x0['x'].dims, xb)), yv,
                    jnp.zeros((T, N), int), jnp.ones(T, bool), m,
                    obs_window_indices=[0, 5, 10])['x'].data
        args = (x0['x'].data, jnp.zeros((T, N)), mask)
        return _shapes_in_jaxpr(jax.make_jaxpr(f)(*args).jaxpr, set())
    fun_shapes = traced(op)
    assert (N, D) not in fun_shapes and (D, N) not in fun_shapes, \
        sorted(fun_shapes)
    # sanity: the matrix path does carry it
    assert (N, D) in traced(jnp.asarray(H)) | traced(jnp.asarray(H))


# ── diagonal R stays a vector ───────────────────────────────────────────────

def test_var4d_operator_window_has_no_NN_R(nature, ops):
    """Default (obs_error_sd) R: no (N, N) array in a traced window, and the
    vector R^{-1} equals the legacy diag(inv(identity * sd^2))."""
    from dabench.dacycler._per_time_h import r_inv_diagonal
    sd = 0.1 + jnp.arange(N) / 50.0
    legacy = jnp.diag(jnp.linalg.inv(jnp.identity(N) * sd ** 2))
    np.testing.assert_allclose(r_inv_diagonal(None, N, sd), legacy,
                               rtol=1e-15)
    np.testing.assert_allclose(r_inv_diagonal(sd ** 2, N, None), legacy,
                               rtol=1e-15)
    np.testing.assert_allclose(r_inv_diagonal(jnp.diag(sd ** 2), N, None),
                               legacy, rtol=1e-15)
    op, H = ops
    obs = _pool_obs(nature, H, stationary=False)
    T = 3
    x0 = nature.isel(time=0)
    mask = jnp.asarray(~np.isnan(np.asarray(obs['x'].data[:T])))
    for Hop in (op, jnp.asarray(H)):
        dc = _var_cycler(Var4DOperator, Hop)
        dc._obs_vector = obs.fillna(0)
        dc.obs_error_sd = OBS_SD
        dc._return_metrics = True
        dc._metrics_mode = "default"

        def f(xb, yv, m):
            out = dc._cycle_obsop(
                    x0.assign(x=(x0['x'].dims, xb)), yv,
                    jnp.zeros((T, N), int), jnp.ones(T, bool), m,
                    obs_window_indices=[0, 5, 10])
            return out[0]['x'].data, out[1]
        shapes = _shapes_in_jaxpr(jax.make_jaxpr(f)(
            x0['x'].data, jnp.zeros((T, N)), mask).jaxpr, set())
        assert (N, N) not in shapes, sorted(shapes)


def test_etkf4d_letkf4d_analysis_has_no_NN_R(ops):
    import xarray as xr
    from dabench.dacycler import ETKF4D
    op, H = ops
    K, T, steps = 5, 3, 4
    owi = jnp.array([0, 2, 3])
    rng = np.random.default_rng(0)
    fc = xr.Dataset({'x': (('ensemble', 'time', 'index'),
                           rng.standard_normal((K, steps, D)))})
    sd = 0.1 + jnp.arange(N) / 50.0
    NT = T * N
    for cls, kw in ((ETKF4D, {}), (LETKF4D, {"localize_radius": 1.5})):
        dc = cls(system_dim=D, delta_t=0.01, ensemble_dim=K, model_obj=None,
                 H=op, **kw)
        dc.obs_error_sd = sd

        def f(y, m):
            Xtraj, Yb, Y, rinv = dc._build_yb(
                    fc, y, jnp.zeros((T, N), int), jnp.ones(T, bool), m,
                    owi)
            Xb = Xtraj[:, 0, :].T
            if cls is ETKF4D:
                return dc._compute_analysis_4d(Xb, Yb, Y, rinv)
            locs = jnp.tile(jnp.arange(N) % D, T)
            return dc._localized_analysis(Xb, Yb, Y, rinv, locs, rho=1.0)
        args = (jnp.asarray(rng.standard_normal((T, N))),
                jnp.ones((T, N), bool))
        Xa = f(*args)
        assert np.all(np.isfinite(np.asarray(Xa)))
        shapes = _shapes_in_jaxpr(jax.make_jaxpr(f)(*args).jaxpr, set())
        assert (N, N) not in shapes and (NT, NT) not in shapes, \
            (cls.__name__, sorted(shapes))
