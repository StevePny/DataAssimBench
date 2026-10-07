"""Weak-constraint (model-error forcing) Var 4D Backprop cycler"""

from typing import Callable

import jax
import jax.numpy as jnp
import jax.scipy as jscipy
import numpy as np
import optax
import xarray as xr
from dabench import _xarray_jax as xj

import dabench.dacycler._utils as dac_utils
from dabench.dacycler._var4d_backprop import Var4DBackprop
from dabench.dacycler._var4d_weak_utils import (
    QHalfLike,
    build_Q_half,
    check_eta_mode,
    expand_eta,
    forced_rollout,
    n_eta,
    )


# For typing
ArrayLike = np.ndarray | jax.Array
XarrayDatasetLike = xr.Dataset | xj.XjDataset


class Var4DBackpropWC(Var4DBackprop):
    """Weak-constraint backpropagation 4D-Var DA Cycler.

    Companion to :class:`Var4DBackprop` adding optional model-error
    control variables (forcing formulation).  With ``Q_half=None``
    (default) every call is delegated to :class:`Var4DBackprop`
    unchanged, so the strong-constraint behaviour is byte-identical.

    Formulation: the window trajectory is ``x_{k+1} = M(x_k) + eta_k``
    (``k = 0 .. T-1``, ``T = steps_per_window - 1``; stepped one model
    step at a time via ``model_obj.forecast(.., n_steps=2)``) and the
    full nonlinear cost minimised by preconditioned SGD is (same factor-
    2 convention as :class:`Var4DBackprop`)

        J = db0^T B^-1 db0 + sum_k chi_k^T chi_k
            + sum_i (y_i - H_i x_{j(i)})^T R^-1 (y_i - H_i x_{j(i)}),

    with ``eta_k = Q^(1/2) chi_k``.  The ``x0`` gradient is
    preconditioned with ``(B^-1 + H^T R^-1 H)^-1`` exactly as in the
    parent; the ``chi`` gradient with the inverse of the approximate
    Gauss-Newton Hessian ``I + sum_i (c_i c_i^T) kron (Q^(1/2)T H_i^T R^-1
    H_i Q^(1/2))`` (dynamics ignored: ``dx_j ~ sum_{k<j} eta_k``, so
    ``c_ik = j(i)`` for a constant forcing and ``1[k < j(i)]`` per step;
    a dense ``(n_eta D)^2`` matrix, built once per window), so
    ``learning_rate=0.5`` is again a Newton-like step.

    Args:
        Q_half: Optional ``Q^(1/2)``. ``None`` (default): strong
            constraint (delegates to the parent).  A scalar ``sigma_q``,
            1-D diagonal ``(system_dim,)``, square factor matrix
            ``(system_dim, system_dim)``, :class:`BFactors`, or a linear
            callable ``chi -> eta``.  (``B`` here is a dense covariance;
            ``Q`` is given as a square root so tiny / rank-deficient Q
            needs no inverse.)
        eta_mode: ``"constant"`` (default; one forcing shared by all
            steps of the window, ECMWF practice) or ``"per_step"``.
        **kwargs: All other arguments as :class:`Var4DBackprop`.
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
        if self._apply_Q_half is None:
            return super()._cycle_obsop(
                    xb0_ds, obs_values, obs_loc_indices, obs_time_mask,
                    obs_loc_mask, H=H, h=h, R=R, B=B,
                    obs_window_indices=obs_window_indices)

        # H / R / B resolution as in Var4DBackprop._cycle_obsop.
        Hs = H
        if H is None and h is None:
            if self.H is None:
                if self.h is not None:
                    raise ValueError(
                            "Var4DBackpropWC only supports linear H.")
                H = self._calc_default_H(obs_loc_indices)
                Hs = jax.lax.cond(
                        self._obs_vector.stationary_observers,
                        lambda: H,
                        lambda: (obs_loc_mask[:, :, jnp.newaxis] * H))
            else:
                H = self.H[jnp.newaxis]
                Hs = jax.lax.cond(
                        self._obs_vector.stationary_observers,
                        lambda: jnp.repeat(H, obs_values.shape[0], axis=0),
                        lambda: (obs_loc_mask[:, :, jnp.newaxis] * H))
        if R is None:
            R = (self._calc_default_R(obs_values, self.obs_error_sd)
                 if self.R is None else self.R)
        if B is None:
            B = self._calc_default_B() if self.B is None else self.B
        Rinv = jscipy.linalg.inv(R)
        Binv = jscipy.linalg.inv(B)
        H0 = Hs.at[0].get()
        hessian_inv = jscipy.linalg.inv(Binv + H0.T @ Rinv @ H0)

        xb0_xr = (xb0_ds.to_xarray()
                  if isinstance(xb0_ds, xj.XjDataset) else xb0_ds)
        stacked_t = xb0_xr.to_stacked_array('system', [])
        x_b0 = jnp.asarray(stacked_t.data).ravel()
        D = x_b0.shape[0]
        T = int(self.steps_per_window) - 1
        owi = jnp.asarray(obs_window_indices)
        n_e = n_eta(self.eta_mode, T + 1)
        Hs_f = jnp.asarray(Hs, x_b0.dtype)

        def to_ds(x):
            return stacked_t.copy(data=x).to_unstacked_dataset(
                    'system').assign_attrs(xb0_xr.attrs)

        def step_fn(x):
            X = self._step_forecast(to_ds(x), 2)[1]
            return jnp.asarray(
                    X.to_stacked_array('system', ['time']).data)[-1]

        def traj(x0, chi):
            eta = expand_eta(jax.vmap(self._apply_Q_half)(chi), T)
            return forced_rollout(step_fn, x0, eta)

        mask = jnp.asarray(obs_time_mask, x_b0.dtype)

        def loss(params):
            x0, chi = params
            X = traj(x0, chi)
            resid = jax.vmap(
                    lambda i: obs_values[i] - Hs_f[i] @ X[owi[i]])(
                    jnp.arange(owi.shape[0]))
            obs_term = jnp.sum(mask * jnp.einsum(
                    'io,op,ip->i', resid, Rinv, resid))
            db0 = x0 - x_b0
            loss_val = db0 @ Binv @ db0 + jnp.sum(chi * chi) + obs_term
            if not self.raise_on_diverge:
                return loss_val
            return jax.lax.cond(
                    jnp.isnan(loss_val),
                    lambda: self._callback_raise_error(
                        self._raise_nan_error, loss_val),
                    lambda: loss_val)

        # Approximate GN preconditioner for chi (dynamics ignored: the
        # forcing accumulates additively, dx_j ~ sum_{k<j} eta_k).
        Qh = jax.vmap(self._apply_Q_half)(
                jnp.eye(D, dtype=x_b0.dtype)).T          # columns = Q^(1/2) e_d
        HtRH = jnp.einsum('iod,op,ipe->ide', Hs_f, Rinv, Hs_f)
        A = jnp.einsum('dp,ide,eq->ipq', Qh, HtRH, Qh)   # Qh^T H^T R^-1 H Qh
        if self.eta_mode == "constant":
            C = owi.astype(x_b0.dtype)[None, :]          # (1, n_obs)
        else:
            C = (jnp.arange(n_e)[:, None] < owi[None, :]).astype(x_b0.dtype)
        hess_chi = (jnp.eye(n_e * D, dtype=x_b0.dtype)
                    + jnp.einsum('ki,li,ipq->kplq', C * mask[None, :], C, A
                                 ).reshape(n_e * D, n_e * D))
        P_chi = jnp.linalg.inv(hess_chi)

        lr = optax.exponential_decay(self.learning_rate, 1, self.lr_decay)
        optimizer = optax.sgd(lr)
        params0 = (x_b0, jnp.zeros((n_e, D), dtype=x_b0.dtype))
        opt_state = optimizer.init(params0)
        vg = jax.value_and_grad(loss)

        def epoch(carry, i):
            params, init_loss, opt_state = carry
            loss_val, (gx, gchi) = vg(params)
            g = (hessian_inv @ gx, (P_chi @ gchi.ravel()).reshape(gchi.shape))
            init_loss = jax.lax.cond(i == 0, lambda: loss_val,
                                     lambda: init_loss)
            if self.raise_on_diverge:
                loss_val = jax.lax.cond(
                        loss_val / init_loss > self.loss_growth_limit,
                        lambda: self._callback_raise_error(
                            self._raise_loss_growth_error, loss_val),
                        lambda: loss_val)
            updates, opt_state = optimizer.update(g, opt_state)
            updates = jax.tree_util.tree_map(
                    lambda u: self.lr_scale * u, updates)
            params = optax.apply_updates(params, updates)
            return (params, init_loss, opt_state), loss_val

        (params, _, _), _ = jax.lax.scan(
                epoch, (params0, jnp.asarray(0., x_b0.dtype), opt_state),
                jnp.arange(self.num_iters))
        x_a, chi_a = params
        xa0_ds = to_ds(x_a)
        if not self._return_metrics:
            return xa0_ds

        # Metrics: O-A against the forced analysis trajectory; end-of-window
        # O-A against the UNFORCED re-forecast (next-cycle IC).
        dtype = jnp.asarray(obs_values).dtype
        Xb = traj(x_b0, jnp.zeros_like(chi_a))
        Xa = traj(x_a, chi_a)
        _, Xa_ds = self.model_obj.forecast(
                xa0_ds, n_steps=self.steps_per_window)
        Xa_free = jnp.asarray(Xa_ds.to_stacked_array('system', ['time']).data)
        Hs_m = jnp.asarray(Hs, dtype)
        idx = jnp.arange(Hs_m.shape[0])
        Hxb = jax.vmap(lambda i: Hs_m[i] @ Xb[owi[i]])(idx)
        Hxa = jax.vmap(lambda i: Hs_m[i] @ Xa[owi[i]])(idx)
        y = jnp.asarray(obs_values, dtype).reshape(-1)
        obs_dim = Hs_m.shape[1]
        active = (jnp.repeat(jnp.asarray(obs_time_mask, bool), obs_dim)
                  & jnp.asarray(obs_loc_mask, bool).reshape(-1))
        sigma2_diag = jnp.broadcast_to(
                jnp.diag(jnp.asarray(R, dtype)),
                (Hs_m.shape[0], obs_dim)).reshape(-1)
        end_idx = self.steps_per_window - 1
        Hxa_end = jax.vmap(lambda i: Hs_m[i] @ Xa_free[end_idx])(idx)
        end_active = (active & jnp.repeat(owi == end_idx, obs_dim))
        metrics = dac_utils._obs_space_metrics(
                y, Hxb.reshape(-1).astype(dtype),
                Hxa.reshape(-1).astype(dtype),
                active, sigma2_diag, ens_obs=None,
                return_per_obs=(self._metrics_mode == "debug"), dtype=dtype,
                Hxa_end_mean=Hxa_end.reshape(-1).astype(dtype),
                end_active_mask=end_active)
        return xa0_ds, metrics
