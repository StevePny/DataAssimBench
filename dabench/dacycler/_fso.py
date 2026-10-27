"""Forecast sensitivity to observations (FSO, Langland and Baker 2004) for the matrix-free incremental 4D-Var.

Let J_e(x_a) be a scalar forecast-error measure chosen by the user (e.g. the misfit of the forecast from the window analysis x_a to verifying data), x_a = x_b + B^(1/2) v_a, and v_a = K d the analysis of the incremental problem linearised about the analysis trajectory:
    A v = L^T R^-1 d,  A = I + L^T R^-1 L,  L v = (H_i dx_(j(i)))_i with dx_0 = B^(1/2) v propagated by the tangent-linear model,
    d_i = y_i - H_i(x_b trajectory)_(j(i)) the innovations of the background trajectory.
The first-order change of J_e due to the observations is  g . (x_a - x_b) = <s, d>  with  g = grad J_e(x_a)  (the adjoint of the forecast, by autodiff) and the observation-space sensitivity
    s = R^-1 L A^-1 B^(1/2)^T g,
so that the impact of observation i is  s_i d_i  (negative = reduces the forecast error): sum_i s_i d_i = g . (x_a - x_b) up to the linearisation (exact for a linear model and observation operator and one outer loop). A is the Hessian of the inner problem and is inverted by the conjugate-gradient solver, one extra solve with the operator that the cycler already uses. s_i is also the derivative of J_e with respect to the observation value y_i (the sensitivity of the forecast
to that observation), which is what the finite-difference tests check.
"""
import jax
import jax.numpy as jnp
from jax.scipy.sparse.linalg import cg


def fso_impacts(L, Rinv, d, g_v, n_cg=500, tol=1e-12):
    """Impacts of every observation: ``L(v) -> (n_obs, obs_dim)`` the linear map of the control vector to the observation-space increments (masked entries zero), ``Rinv`` the weights ``1 / sd^2`` (same shape), ``d`` the innovations (same shape, zero where
    there is no observation), ``g_v`` = B^(1/2)^T g the forecast-error gradient in control space. Returns ``dict(impact, sensitivity, innovation, total)`` with ``total = sum(impact)``."""
    v0 = jnp.zeros_like(g_v)
    _, Lt = jax.vjp(L, v0)

    def A(v):
        return v + Lt(Rinv * L(v))[0]

    z, _ = cg(A, g_v, tol=tol, maxiter=n_cg)
    s = Rinv * L(z)
    imp = s * d
    return dict(impact=imp, sensitivity=s, innovation=d, total=jnp.sum(imp))
