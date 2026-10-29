"""Weak-constraint (model-error forcing) matrix-free 4D-Var cycler"""

from typing import Callable

import jax
import jax.numpy as jnp
import jax.scipy as jscipy
import numpy as np
import xarray as xr
from dabench import _xarray_jax as xj

import dabench.dacycler._utils as dac_utils
from dabench.dacycler._per_time_h import (make_per_time_H, as_dtype,
                                          r_inv_diagonal)
from dabench.dacycler._var4d_operator import Var4DOperator
from dabench.dacycler._var4d_operator_utils import pcg_lanczos_solve
from dabench.dacycler._var4d_weak_utils import (
    QHalfLike,
    build_Q_half,
    check_eta_mode,
    expand_eta,
    forced_rollout,
    model_rollout,
    n_eta,
    quadratic_cost_wc,
    window_tlm_rollout_forced,
    )


_ETA_ATTR = "_wc_eta_seq"

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
    ``x_a = x_b + B^(1/2) v_total`` (as in :class:`Var4DOperator`).  With
    ``forecast_with_eta=True`` (default) the cycler's subsequent forecast
    (and so the next window's background) is the *forced* model that the
    analysis was fitted with: the estimated ``eta_k`` over the window, and
    beyond the window the window-mean ``eta`` held constant.  With
    ``forecast_with_eta=False`` the forecast uses the unforced model (the
    previous behaviour).

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
                 forecast_with_eta: bool = True,
                 **kwargs):
        self.hvp_mode = kwargs.pop('hvp_mode', 'gn')  # 'gn': Hessian-vector product as one gradient of the quadratic part; 'jvp': forward-over-reverse derivative of the gradient (previous behaviour)
        super().__init__(*args, **kwargs)
        self.eta_mode = check_eta_mode(eta_mode)
        self.forecast_with_eta = bool(forecast_with_eta)
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
                # Single H (obs_dim, system_dim); per-time mask applied to the
                # product, never materialised as (T, obs_dim, system_dim).
                Hs = make_per_time_H(
                        self.H, obs_values.shape[0], obs_loc_mask,
                        self._obs_vector.stationary_observers)
        if R is None:
            R = self.R          # None -> diagonal from obs_error_sd (vector)
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
                         obs_time_mask, R_inv_diag, outer_idx=0, aux=None):
        """Joint PCG-Lanczos inner solve over ``(delta_v, delta_chi)``."""
        innovations = jax.vmap(
                lambda i: obs_vals[i] - Hs[i] @ x_l_traj[obs_window_indices[i]]
                )(jnp.arange(Hs.shape[0]))
        D = v_total.shape[0]
        chi_shape = chi_total.shape
        kw_aux = {} if aux is None else dict(aux=aux)

        def J_of(z: ArrayLike) -> jax.Array:
            return quadratic_cost_wc(
                    z[:D], z[D:].reshape(chi_shape), v_total, chi_total,
                    tlm_op=tlm_op, x_l_traj=x_l_traj, Hs=Hs,
                    innovations=innovations,
                    obs_window_indices=obs_window_indices,
                    obs_time_mask=obs_time_mask, R_inv_diag=R_inv_diag,
                    apply_B_half=apply_B_half,
                    apply_Q_half=self._apply_Q_half, **kw_aux)

        grad_J = jax.grad(J_of)
        z0 = jnp.zeros(D + chi_total.size, dtype=v_total.dtype)
        b_rhs = -grad_J(z0)
        lm = self.lm_lambda

        if self.hvp_mode == "gn":
            # The cost is exactly quadratic, so its Hessian is the (Gauss-Newton)
            # matrix H of the pure quadratic part: grad of the cost without the
            # linear terms (offsets v_total, chi_total and innovations set to
            # zero) is H z.  One reverse pass over the tangent-linear window
            # replaces the forward-over-reverse derivative of the gradient.
            def J_quad(z: ArrayLike) -> jax.Array:
                return quadratic_cost_wc(
                        z[:D], z[D:].reshape(chi_shape),
                        jnp.zeros_like(v_total), jnp.zeros_like(chi_total),
                        tlm_op=tlm_op, x_l_traj=x_l_traj, Hs=Hs,
                        innovations=jnp.zeros_like(innovations),
                        obs_window_indices=obs_window_indices,
                        obs_time_mask=obs_time_mask, R_inv_diag=R_inv_diag,
                        apply_B_half=apply_B_half,
                        apply_Q_half=self._apply_Q_half, **kw_aux)

            grad_Jq = jax.grad(J_quad)

            def hvp_fn(p: ArrayLike) -> jax.Array:
                return grad_Jq(p) + lm * p
        else:
            def hvp_fn(p: ArrayLike) -> jax.Array:
                return jax.jvp(grad_J, (z0,), (p,))[1] + lm * p

        dz, info = pcg_lanczos_solve(
                hvp_fn, b_rhs, max_iter=self.n_inner_loops,
                tol=self.inner_tol, verbose=self.verbose,
                verbose_tag=f"pcg-wc/outer{int(outer_idx)}")
        if self.health:
            # nonlinear cost at this outer's iterate (its trajectory x_l_traj) and the quadratic model's value after the inner solve
            w = jnp.broadcast_to(R_inv_diag, innovations.shape)
            act = obs_time_mask[:, None] & (w > 0) & (innovations != 0)  # dense observation vectors: unobserved elements carry a zero innovation, they are not observations
            J_o = 0.5 * jnp.sum(jnp.where(act, w * innovations * innovations, 0.0))
            n_o = jnp.sum(act)
            J_b = 0.5 * jnp.sum(v_total * v_total)
            J_q = 0.5 * jnp.sum(chi_total * chi_total)
            J0 = J_b + J_q + J_o
            J1 = J_of(dz)
            jax.debug.print(
                    "[4dvar-health] outer {o}: J {J:.4e} = Jb {Jb:.3e} + Jq {Jq:.3e} + Jo {Jo:.3e} "
                    "(2Jo/p {chi:.3f}, p {p}) | after inner: J {J1:.4e} (ratio {r:.3f}) | "
                    "|grad| {g:.3e} | PCG {n} it, converged {c}",
                    o=int(outer_idx), J=J0, Jb=J_b, Jq=J_q, Jo=J_o,
                    chi=2 * J_o / jnp.maximum(n_o, 1), p=n_o, J1=J1,
                    r=J1 / jnp.where(J0 > 0, J0, 1.0),
                    g=jnp.linalg.norm(b_rhs), n=info["n_iter"],
                    c=info["converged"])
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

        if jnp.ndim(self.obs_error_sd) == 2:
            raise NotImplementedError(
                "an error per observation (2-D obs_error_sd) needs the "
                "compiled-window path (model_inputs, obs_inputs or "
                "compile_once)")
        Hs, R = self._resolve_Hs_R(obs_values, obs_loc_indices,
                                   obs_loc_mask, H, h, R)
        R_inv_diag = r_inv_diagonal(R, obs_values.shape[-1],
                                    self.obs_error_sd)
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
        if self.forecast_with_eta:
            # carried to _step_forecast in the same trace (not part of any
            # scan carry); stripped there before the model sees the state
            xa_ds = xa_ds.assign_attrs({_ETA_ATTR: eta_seq(chi_total)})
        if not self._return_metrics:
            return xa_ds

        # Metrics: O-A uses the forced analysis trajectory (what the
        # analysis fits); the end-of-window O-A uses the trajectory the next
        # cycle starts from (forced unless forecast_with_eta=False).
        dtype = x_a.dtype
        xa_traj = forced_rollout(step_fn, x_a, eta_seq(chi_total))
        xa_free = (xa_traj if self.forecast_with_eta
                   else self._rollout_background(
                       xa_ds, n_steps=self.steps_per_window))
        Hs_m = as_dtype(Hs, dtype)
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

    def fso(self, input_state, start_time, obs_vector, obs_error_sd,
            analysis_window, forecast_error, model_inputs=None,
            obs_inputs=None, obs_times_per_window=None,
            analysis_time_in_window=None, cycle=0, n_cg=500):
        """Forecast sensitivity to observations of one window with the
        weak-constraint controls ``(v, chi)`` (see
        :mod:`dabench.dacycler._fso`).  ``forecast_error(x_a, eta_seq)`` is
        a scalar function of the analysis state at the window start and of
        the model-error forcing sequence ``eta_seq`` (``(steps, system_dim)``,
        the estimated forcing the forecast carries), e.g. the misfit of a
        forecast made with them.  The control vector is the stacked ``v`` and
        ``chi``; the sensitivities and impacts have the window's observation
        shape ``(obs_times, obs_dim)`` and the impacts add up to
        ``grad(J_e) . (z_a)`` mapped to the state and forcing
        (``total_linear``).  Without ``Q_half`` this is the strong-constraint
        :meth:`Var4DOperator.fso`.  Compiled-window path."""
        if self._apply_Q_half is None:
            return super().fso(
                    input_state, start_time, obs_vector, obs_error_sd,
                    analysis_window, lambda x_a: forecast_error(x_a, None),
                    model_inputs, obs_inputs, obs_times_per_window,
                    analysis_time_in_window, cycle, n_cg)
        from dabench.dacycler._fso import fso_impacts
        self._check_B_compiled()
        self._return_metrics = False
        setup = self._compiled_setup(
            input_state, start_time, obs_vector, int(cycle) + 1, obs_error_sd,
            analysis_window, analysis_time_in_window, model_inputs,
            obs_times_per_window)
        window = self._compiled_window()
        x_b = setup['x_b']
        for c in range(int(cycle)):
            args = self._window_args(setup, c, start_time, analysis_window,
                                     model_inputs, obs_inputs)
            _, traj = window(x_b, *args)
            x_b = traj[-1]
        (obs_vals, loc_idx, loc_mask, time_mask, obs_times, sd, cur_time, u,
         o) = self._window_args(setup, int(cycle), start_time,
                                analysis_window, model_inputs, obs_inputs)
        S = int(self.steps_per_window)
        T = S - 1
        times = cur_time + jnp.arange(T) * self.delta_t
        owi = jnp.argmin(jnp.abs(obs_times[:, None]
                                 - (cur_time + self._model_timesteps)[None]),
                         axis=1)
        Hs = self._resolve_Hs_inputs(obs_vals, loc_idx, loc_mask, o)
        Rinv = r_inv_diagonal(self.R, obs_vals.shape[-1], sd)
        F = self._model_step_inputs()
        aux = (times, u)
        u0 = jax.tree_util.tree_map(lambda a: a[0], u)
        D = x_b.shape[0]
        v_total = jnp.zeros_like(x_b)
        chi_total = jnp.zeros((n_eta(self.eta_mode, T + 1), D),
                              dtype=x_b.dtype)
        apply_B_half = None

        def eta_seq(chi):
            return expand_eta(jax.vmap(self._apply_Q_half)(chi), T)

        for outer in range(self.n_outer_loops):    # the analysis, as the window
            x_l0 = (x_b if apply_B_half is None
                    else x_b + apply_B_half(v_total))
            x_l_traj = forced_rollout(F, x_l0, eta_seq(chi_total),
                                      times=times, inputs=u)
            tlm_op = self._tlm_op_inputs(x_l_traj[0])
            apply_B_half = self._ensure_B_half(
                    x_l_traj[0],
                    lambda x, dx: tlm_op(x, dx, times[0], u0), outer)
            dv, dchi, _ = self._innerloop_4d_wc(
                    x_l_traj, v_total, chi_total, tlm_op, apply_B_half,
                    Hs, obs_vals, owi, time_mask,
                    r_inv_diagonal(self.R, obs_vals.shape[-1], sd),
                    outer_idx=outer, aux=aux)
            v_total = v_total + dv
            chi_total = chi_total + dchi
        x_a = x_b + apply_B_half(v_total)
        # linearise about the analysis trajectory (forced with the fitted eta)
        xa_traj = forced_rollout(F, x_a, eta_seq(chi_total), times=times,
                                 inputs=u)
        tlm_op = self._tlm_op_inputs(xa_traj[0])
        n_obs = obs_vals.shape[0]
        mask = (jnp.asarray(time_mask)[:, None] & jnp.asarray(loc_mask)
                & jnp.isfinite(obs_vals))
        zero_eta = jnp.zeros((T, D), dtype=x_b.dtype)
        xb_traj = forced_rollout(F, x_b, zero_eta, times=times, inputs=u)
        Hxb = jax.vmap(lambda i: Hs[i] @ xb_traj[owi[i]])(jnp.arange(n_obs))
        d = jnp.where(mask, obs_vals - Hxb, 0.0)
        Rinv = jnp.broadcast_to(Rinv, d.shape) * mask
        chi_shape = chi_total.shape
        nv = D

        def L(z):
            dx0 = apply_B_half(z[:nv])
            deta = expand_eta(jax.vmap(self._apply_Q_half)(
                    z[nv:].reshape(chi_shape)), T)
            dxs = window_tlm_rollout_forced(tlm_op, xa_traj, dx0, deta,
                                            aux=aux)
            return jax.vmap(lambda i: Hs[i] @ dxs[owi[i]])(jnp.arange(n_obs)) * mask

        eta_a = eta_seq(chi_total)
        gx, ge = jax.grad(forecast_error, argnums=(0, 1))(x_a, eta_a)
        _, Bt = jax.vjp(apply_B_half, jnp.zeros_like(x_a))
        _, Qt = jax.vjp(lambda c: expand_eta(
                jax.vmap(self._apply_Q_half)(c), T), jnp.zeros(chi_shape,
                                                               x_b.dtype))
        g_z = jnp.concatenate([Bt(gx)[0], Qt(ge)[0].ravel()])
        res = fso_impacts(L, Rinv, d, g_z, n_cg=n_cg)
        z_a = jnp.concatenate([v_total, chi_total.ravel()])
        # first-order change of J_e by the analysis increment and the fitted forcing
        res.update(total_linear=jnp.vdot(gx, x_a - x_b) + jnp.vdot(ge, eta_a),
                   x_a=x_a, x_b=x_b, eta_seq=eta_a, z_a=z_a)
        return res

    def _window_inputs(self, x_b0, obs_vals, obs_loc_indices, obs_loc_mask,
                       obs_time_mask, obs_times, obs_error_sd, cur_time, u,
                       o):
        """Compiled-window path, weak constraint: the forced window (as
        :meth:`_cycle_obsop`) with the model time and inputs threaded through
        the rollouts and the TLM; the returned trajectory is the forecast over
        the window from ``x_a``, forced with the fitted ``eta_k`` when
        ``forecast_with_eta`` (as :meth:`_step_forecast`)."""
        if self._apply_Q_half is None:
            return super()._window_inputs(
                    x_b0, obs_vals, obs_loc_indices, obs_loc_mask,
                    obs_time_mask, obs_times, obs_error_sd, cur_time, u, o)
        S = int(self.steps_per_window)
        T = S - 1
        times = cur_time + jnp.arange(T) * self.delta_t
        owi = jnp.argmin(jnp.abs(obs_times[:, None]
                                 - (cur_time + self._model_timesteps)[None]),
                         axis=1)
        Hs = self._resolve_Hs_inputs(obs_vals, obs_loc_indices,
                                     obs_loc_mask, o)
        R_inv_diag = r_inv_diagonal(self.R, obs_vals.shape[-1], obs_error_sd)
        F = self._model_step_inputs()
        aux = (times, u)
        u0 = jax.tree_util.tree_map(lambda a: a[0], u)
        D = x_b0.shape[0]
        v_total = jnp.zeros_like(x_b0)
        chi_total = jnp.zeros((n_eta(self.eta_mode, T + 1), D),
                              dtype=x_b0.dtype)
        apply_B_half = None

        def eta_seq(chi):
            return expand_eta(jax.vmap(self._apply_Q_half)(chi), T)

        for outer in range(self.n_outer_loops):
            x_l0 = (x_b0 if apply_B_half is None
                    else x_b0 + apply_B_half(v_total))
            x_l_traj = forced_rollout(F, x_l0, eta_seq(chi_total),
                                      times=times, inputs=u)
            tlm_op = self._tlm_op_inputs(x_l_traj[0])
            apply_B_half = self._ensure_B_half(
                    x_l_traj[0],
                    lambda x, dx: tlm_op(x, dx, times[0], u0), outer)
            dv, dchi, info = self._innerloop_4d_wc(
                    x_l_traj, v_total, chi_total, tlm_op, apply_B_half,
                    Hs, obs_vals, owi, obs_time_mask, R_inv_diag,
                    outer_idx=outer, aux=aux)
            v_total = v_total + dv
            chi_total = chi_total + dchi
        x_a = x_b0 + apply_B_half(v_total)
        if self.forecast_with_eta:
            traj = forced_rollout(F, x_a, eta_seq(chi_total), times=times,
                                  inputs=u)
        else:
            traj = model_rollout(F, x_a, times, u)
        return x_a, traj

    def _step_forecast(self,
                       xa: XarrayDatasetLike,
                       n_steps: int = 1):
        """Forecast with the model the analysis was fitted with.

        Without a carried ``eta`` (strong constraint, or
        ``forecast_with_eta=False``) this is the parent's unforced
        ``model_obj.forecast``.  Otherwise the trajectory is
        ``x_{k+1} = M(x_k) + eta_k`` with the window's ``eta_k`` and, past
        the window, their mean.  Returns ``(last frame, all frames)`` as
        Datasets with a ``time`` dimension, like ``model_obj.forecast``.
        """
        xa_xr = xa.to_xarray() if isinstance(xa, xj.XjDataset) else xa
        eta = xa_xr.attrs.get(_ETA_ATTR, None)
        if eta is None:
            return super()._step_forecast(xa, n_steps=n_steps)
        attrs = {k: v for k, v in xa_xr.attrs.items() if k != _ETA_ATTR}
        xa_clean = xa_xr.assign_attrs(attrs)
        xa_clean.attrs = attrs
        x0 = jnp.asarray(xa_clean.to_stacked_array('system', []).data).ravel()
        n = int(n_steps) - 1
        T = eta.shape[0]
        if n <= T:
            seq = eta[:n]
        else:
            seq = jnp.concatenate(
                [eta, jnp.broadcast_to(eta.mean(axis=0), (n - T,) + eta.shape[1:])],
                axis=0)
        traj = forced_rollout(self._make_step_fn(xa_clean), x0, seq)
        st = xa_clean.to_stacked_array('system', [])
        dt = float(getattr(self.model_obj, 'delta_t', 1.0))
        da = xr.DataArray(traj, dims=('time', 'system'),
                          coords={'system': st['system'],
                                  'time': np.arange(int(n_steps)) * dt})
        ds = da.to_unstacked_dataset('system').assign_attrs(attrs)
        return ds.isel(time=-1), ds
