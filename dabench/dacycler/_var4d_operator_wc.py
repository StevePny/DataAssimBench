"""Weak-constraint (model-error forcing) matrix-free 4D-Var cycler"""

from typing import Callable

import jax
import jax.numpy as jnp
import jax.scipy as jscipy
import numpy as np
import xarray as xr
from dabench import _xarray_jax as xj

import dabench.dacycler._utils as dac_utils
from dabench.dacycler._var4d_operator import Var4DOperator
from dabench.dacycler._var4d_operator_utils import pcg_lanczos_solve
from dabench.dacycler._var4d_weak_utils import (
    QHalfLike,
    build_Q_half,
    check_eta_mode,
    expand_eta,
    forced_rollout,
    n_eta,
    quadratic_cost_wc,
    )


# For typing
ArrayLike = np.ndarray | jax.Array
XarrayDatasetLike = xr.Dataset | xj.XjDataset


class Var4DOperatorWC(Var4DOperator):
    """Matrix-free weak-constraint 4D-Var DA Cycler.

    Companion to :class:`Var4DOperator` adding optional model-error
    control variables (forcing formulation).  With ``Q_half=None``
    (default) every call is delegated to :class:`Var4DOperator`
    unchanged, so the strong-constraint behaviour is byte-identical.

    Formulation: the window trajectory is
    ``x_{k+1} = M(x_k) + eta_k`` (``k = 0 .. T-1``,
    ``T = steps_per_window - 1``) and the cost is

        J = 1/2 dx0^T B^-1 dx0 + 1/2 sum_k eta_k^T Q^-1 eta_k + J_o.

    Both ``dx0`` and ``eta`` are CVT-preconditioned:
    ``dx0 = B^(1/2) v`` and ``eta_k = Q^(1/2) chi_k``.  Each outer
    Gauss-Newton iteration rolls the *forced* nonlinear trajectory from
    ``x_b + B^(1/2) v_total`` with ``eta = Q^(1/2) chi_total``
    (one model step at a time via ``model_obj.forecast(.., n_steps=2)``),
    closes the TLM, and solves the inner GN system for
    ``(delta_v, delta_chi)`` jointly by PCG-Lanczos on the stacked
    control vector (see :func:`quadratic_cost_wc`; the increment is
    propagated relative to the linearisation trajectory).

    The returned analysis is the window-start state
    ``x_a = x_b + B^(1/2) v_total`` (as in :class:`Var4DOperator`); the
    cycler's subsequent forecast uses the unforced model.

    Args:
        Q_half: Optional model-error covariance square root ``Q^(1/2)``.
            ``None`` (default): strong constraint (delegates to the
            parent).  A scalar ``sigma_q`` (``sigma_q I``), a 1-D
            diagonal ``(system_dim,)``, a square factor matrix
            ``(system_dim, system_dim)``, a :class:`BFactors`, or a
            linear callable ``chi -> eta`` (mirrors ``B_factors`` /
            ``B_half_op``).
        eta_mode: Where ``eta`` enters. ``"constant"`` (default): one
            forcing shared by all steps of the window (bias-like, ECMWF
            practice).  ``"per_step"``: an independent ``eta_k`` per
            model step.
        **kwargs: All other arguments as :class:`Var4DOperator`.
            ``tlm_op`` is reused unchanged for the forced recursion
            ``dx_{k+1} = M_k dx_k + deta_k``.
    """

    def __init__(self,
                 *args,
                 Q_half: QHalfLike = None,
                 eta_mode: str = "constant",
                 **kwargs):
        super().__init__(*args, **kwargs)
        self.eta_mode = check_eta_mode(eta_mode)
        self.Q_half = Q_half
        self._apply_Q_half = build_Q_half(Q_half, int(self.system_dim))

    def _resolve_Hs_R(self, obs_values, obs_loc_indices, obs_loc_mask,
                      H, h, R):
        """Same H / R resolution as :meth:`Var4DOperator._cycle_obsop`."""
        Hs = H
        if H is None and h is None:
            if self.H is None:
                if self.h is None:
                    H = self._calc_default_H(obs_loc_indices)
                    Hs = jax.lax.cond(
                            self._obs_vector.stationary_observers,
                            lambda: H,
                            lambda: (obs_loc_mask[:, :, jnp.newaxis] * H))
                else:
                    raise ValueError(
                            "Var4DOperatorWC only supports linear H.")
            else:
                H = self.H[jnp.newaxis]
                Hs = jax.lax.cond(
                        self._obs_vector.stationary_observers,
                        lambda: jnp.repeat(H, obs_values.shape[0], axis=0),
                        lambda: (obs_loc_mask[:, :, jnp.newaxis] * H))
        if R is None:
            R = (self._calc_default_R(obs_values, self.obs_error_sd)
                 if self.R is None else self.R)
        return Hs, R

    def _make_step_fn(self, template: xr.Dataset
                      ) -> Callable[[ArrayLike], ArrayLike]:
        """One nonlinear model step on a flat state."""
        def step_fn(x: ArrayLike) -> jax.Array:
            x_ds = self._array_to_dataset_like(x, template)
            return self._rollout_background(x_ds, n_steps=2)[-1]
        return step_fn

    def _innerloop_4d_wc(self, x_l_traj, v_total, chi_total, tlm_op,
                         apply_B_half, Hs, obs_vals, obs_window_indices,
                         obs_time_mask, R_inv_diag, outer_idx=0):
        """Joint PCG-Lanczos inner solve over ``(delta_v, delta_chi)``."""
        innovations = jax.vmap(
                lambda i: obs_vals[i] - Hs[i] @ x_l_traj[obs_window_indices[i]]
                )(jnp.arange(Hs.shape[0]))
        D = v_total.shape[0]
        chi_shape = chi_total.shape

        def J_of(z: ArrayLike) -> jax.Array:
            return quadratic_cost_wc(
                    z[:D], z[D:].reshape(chi_shape), v_total, chi_total,
                    tlm_op=tlm_op, x_l_traj=x_l_traj, Hs=Hs,
                    innovations=innovations,
                    obs_window_indices=obs_window_indices,
                    obs_time_mask=obs_time_mask, R_inv_diag=R_inv_diag,
                    apply_B_half=apply_B_half,
                    apply_Q_half=self._apply_Q_half)

        grad_J = jax.grad(J_of)
        z0 = jnp.zeros(D + chi_total.size, dtype=v_total.dtype)
        b_rhs = -grad_J(z0)
        lm = self.lm_lambda

        def hvp_fn(p: ArrayLike) -> jax.Array:
            return jax.jvp(grad_J, (z0,), (p,))[1] + lm * p

        dz, info = pcg_lanczos_solve(
                hvp_fn, b_rhs, max_iter=self.n_inner_loops,
                tol=self.inner_tol, verbose=self.verbose,
                verbose_tag=f"pcg-wc/outer{int(outer_idx)}")
        return dz[:D], dz[D:].reshape(chi_shape), info

    def _cycle_obsop(self,
                     xb0_ds: XarrayDatasetLike,
                     obs_values: ArrayLike,
                     obs_loc_indices: ArrayLike,
                     obs_time_mask: ArrayLike,
                     obs_loc_mask: ArrayLike,
                     H: ArrayLike | None = None,
                     h: Callable | None = None,
                     R: ArrayLike | None = None,
                     B: ArrayLike | None = None,
                     obs_window_indices=None,
                     ) -> XarrayDatasetLike:
        """One window; strong-constraint parent path when ``Q_half`` is None."""
        if self._apply_Q_half is None:
            return super()._cycle_obsop(
                    xb0_ds, obs_values, obs_loc_indices, obs_time_mask,
                    obs_loc_mask, H=H, h=h, R=R, B=B,
                    obs_window_indices=obs_window_indices)

        Hs, R = self._resolve_Hs_R(obs_values, obs_loc_indices,
                                   obs_loc_mask, H, h, R)
        R_inv_diag = jnp.diag(jscipy.linalg.inv(R))
        owi = jnp.asarray(obs_window_indices)

        xb0_xr = (xb0_ds.to_xarray()
                  if isinstance(xb0_ds, xj.XjDataset) else xb0_ds)
        x_b0 = jnp.asarray(
                xb0_xr.to_stacked_array('system', []).data).ravel()
        D = x_b0.shape[0]
        T = int(self.steps_per_window) - 1
        step_fn = self._make_step_fn(xb0_xr)
        v_total = jnp.zeros_like(x_b0)
        chi_total = jnp.zeros((n_eta(self.eta_mode, T + 1), D),
                              dtype=x_b0.dtype)
        apply_B_half = None
        xb_traj0 = None

        def eta_seq(chi):
            return expand_eta(jax.vmap(self._apply_Q_half)(chi), T)

        for outer in range(self.n_outer_loops):
            x_l0 = (x_b0 if apply_B_half is None
                    else x_b0 + apply_B_half(v_total))
            x_l_traj = forced_rollout(step_fn, x_l0, eta_seq(chi_total))
            if outer == 0:
                xb_traj0 = x_l_traj
            tlm_op = self.tlm_op_factory(self.model_obj, x_l_traj[0])
            apply_B_half = self._ensure_B_half(x_l_traj[0], tlm_op, outer)
            dv, dchi, info = self._innerloop_4d_wc(
                    x_l_traj, v_total, chi_total, tlm_op, apply_B_half,
                    Hs, obs_values, owi, obs_time_mask, R_inv_diag,
                    outer_idx=outer)
            v_total = v_total + dv
            chi_total = chi_total + dchi

        x_a = x_b0 + apply_B_half(v_total)
        xa_ds = self._array_to_dataset_like(x_a, xb0_xr)
        if not self._return_metrics:
            return xa_ds

        # Metrics: O-A uses the forced analysis trajectory (what the
        # analysis fits); the end-of-window O-A uses the UNFORCED re-forecast
        # (what the next cycle starts from).
        dtype = x_a.dtype
        xa_traj = forced_rollout(step_fn, x_a, eta_seq(chi_total))
        xa_free = self._rollout_background(xa_ds,
                                           n_steps=self.steps_per_window)
        Hs_m = jnp.asarray(Hs, dtype)
        idx = jnp.arange(Hs_m.shape[0])
        Hxb = jax.vmap(lambda i: Hs_m[i] @ xb_traj0[owi[i]])(idx)
        Hxa = jax.vmap(lambda i: Hs_m[i] @ xa_traj[owi[i]])(idx)
        y = jnp.asarray(obs_values, dtype).reshape(-1)
        obs_dim = Hs_m.shape[1]
        active = (jnp.repeat(jnp.asarray(obs_time_mask, bool), obs_dim)
                  & jnp.asarray(obs_loc_mask, bool).reshape(-1))
        sigma2_loc = jnp.where(R_inv_diag > 0,
                               1.0 / jnp.where(R_inv_diag > 0, R_inv_diag,
                                               jnp.ones_like(R_inv_diag)),
                               jnp.zeros_like(R_inv_diag))
        sigma2_diag = jnp.broadcast_to(
                sigma2_loc.astype(dtype),
                (Hs_m.shape[0], obs_dim)).reshape(-1)
        end_idx = self.steps_per_window - 1
        Hxa_end = jax.vmap(lambda i: Hs_m[i] @ xa_free[end_idx])(idx)
        end_active = (active & jnp.repeat(owi == end_idx, obs_dim))
        metrics = dac_utils._obs_space_metrics(
                y, Hxb.reshape(-1).astype(dtype),
                Hxa.reshape(-1).astype(dtype),
                active, sigma2_diag, ens_obs=None,
                return_per_obs=(self._metrics_mode == "debug"), dtype=dtype,
                Hxa_end_mean=Hxa_end.reshape(-1).astype(dtype),
                end_active_mask=end_active)
        return xa_ds, metrics
