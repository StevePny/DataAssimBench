"""Class for matrix-free (operator-based) Var 4D Data Assimilation Cycler"""

import inspect
import os
from functools import partial
from typing import Callable

import jax
import jax.numpy as jnp
import jax.scipy as jscipy
import numpy as np
import xarray as xr
from dabench import _xarray_jax as xj

from dabench import dacycler
import dabench.dacycler._utils as dac_utils
from dabench.dacycler._per_time_h import (make_per_time_H, as_dtype,
                                          is_function_operator,
                                          r_inv_diagonal)
from dabench.model import Model
from dabench.dacycler._var4d_operator_utils import (
    BFactors,
    build_B_half,
    extract_B_factors,
    pcg_lanczos_solve,
    quadratic_cost,
    )
from dabench.dacycler._var4d_weak_utils import model_rollout


# For typing
ArrayLike = np.ndarray | jax.Array
XarrayDatasetLike = xr.Dataset | xj.XjDataset
TLMOpFactory = Callable[[Model, ArrayLike], Callable[
        [ArrayLike, ArrayLike], ArrayLike]]



def _print_lanczos_kappa(alphas, betas, n_iter, outer):
    """Ritz values of the CG-implied Lanczos tridiagonal:
    T[k,k] = 1/a_k + b_{k-1}/a_{k-1}, T[k,k+1] = sqrt(b_k)/a_k."""
    n = int(n_iter)
    a = np.asarray(alphas, np.float64)[:n]
    b = np.asarray(betas, np.float64)[:n]
    if n < 2 or np.any(a <= 0):
        print(f"[lanczos/outer{outer}] n={n} (no estimate)", flush=True)
        return
    diag = 1.0 / a
    diag[1:] += b[:-1] / a[:-1]
    off = np.sqrt(np.maximum(b[:-1], 0.0)) / a[:-1]
    T = np.diag(diag) + np.diag(off, 1) + np.diag(off, -1)
    ev = np.linalg.eigvalsh(T)
    print(f"[lanczos/outer{outer}] n={n} ritz_min={ev[0]:.4e} "
          f"ritz_max={ev[-1]:.4e} kappa_est={ev[-1] / max(ev[0], 1e-300):.4e}",
          flush=True)

class Var4DOperator(dacycler.DACycler):
    """Matrix-free 4D-Var DA Cycler.

    Mirrors :class:`Var4D` but replaces the explicit-Jacobian path
    (``model_obj.compute_tlm`` returning a stacked ``(T, D, D)`` matrix
    plus ``bicgstab`` on ``I + B sum_t M^T H^T R^{-1} H M``) with a
    matrix-free operator path:

      * The tangent-linear model is provided as a callable
        ``tlm_op(x_t, dx_t) -> dx_{t+1}`` (e.g. produced by a TLM
        emulator's ``jax.vjp``).  No Jacobian is ever materialised.
      * The background-error covariance is supplied either as factors
        ``BFactors(U, sigma, sigma_bg)`` or as a callable
        ``B_half_op(v) -> dx`` and absorbed into the cost via the
        control-variable transform ``dx_0 = B^(1/2) v``.
      * The inner GN normal equation is solved by
        :func:`pcg_lanczos_solve`, which exposes the Lanczos
        tridiagonal / Ritz values as inner-loop diagnostics.

    The cycle / window / observation plumbing is inherited from
    :class:`DACycler`; ``_cycle_obsop`` is the only override required
    by ``cycle()``.  Suitable for high-dimensional ML-emulated TLMs
    where forming ``M`` densely is prohibitive.

    Args:
        system_dim: System dimension.
        delta_t: The timestep of the model (assumed uniform).
        model_obj: Forecast model object. Used only for the
            nonlinear background rollout via
            ``model_obj.forecast(state_vec, n_steps=...)``; the
            Jacobian is never queried.
        tlm_op_factory: Callable
            ``(model_obj, x_linear) -> tlm_op(x_t, dx_t)`` that closes
            the TLM at a specified linearisation state. Called once
            per outer iteration. The returned callable must be
            strictly linear in ``dx_t``.
        tlm_adj_op_factory: Optional adjoint factory with signature
            ``(model_obj, x_linear) -> tlm_adj_op(x_t, w_t)``. When
            None (default), the adjoint is obtained via ``jax.vjp``
            on the forward TLM, which is sufficient for the PCG-
            Lanczos HVP path.
        B_factors: Optional :class:`BFactors` background covariance
            factorisation. If None and ``B_half_op`` is also None,
            factors are estimated via :func:`extract_B_factors` on
            the outer-0 TLM with rank ``B_rank`` and ``B_probes``
            probes (or, if also None, falls back to identity).
        B_half_op: Optional matrix-free ``B^(1/2)`` callable. Takes
            precedence over ``B_factors`` when both are provided.
        B_half_factory: Optional per-cycle ``B^(1/2)`` factory
            ``B_half_factory(x_linear, tlm_op, outer_idx) -> apply_B_half``.
            Rebuilt at outer 0 of every analysis cycle from the current
            background ``x_linear`` (flow-dependent "errors of the day"
            B), then held fixed across the remaining outers of that
            window (unless ``refresh_B_each_outer``).  Used in place of a
            static ``B_factors`` / RSVD; ranks below ``B_half_op`` but
            above ``B_factors`` in the resolution order.
        B_rank: Retained rank for auto-built ``B_factors``. Default 50.
        B_probes: Number of Gaussian probes for the RSVD when
            auto-building. Default 64.
        B_floor: Isotropic floor for auto-built ``B_factors``.
        R: Observation error covariance matrix.
        H: Observation operator matrix.
        h: Callable observation operator (currently unused: only
            linear ``H`` paths are supported, matching :class:`Var4D`).
        n_outer_loops: Outer Gauss-Newton iterations. Default 3.
        n_inner_loops: Inner PCG-Lanczos iterations per outer.
            Default 30.
        inner_tol: Relative residual tolerance for inner solve.
        steps_per_window: Number of timesteps per analysis window.
        obs_window_indices: Trajectory step for each obs in the
            analysis window.
        refresh_B_each_outer: When True, rebuild ``B^(1/2)`` from a
            fresh RSVD at every outer iteration (default False —
            holds ``B`` fixed so Ritz diagnostics are comparable
            across outers).
    """
    _in_4d: bool = True
    _uses_ensemble: bool = False

    def __init__(self,
                 system_dim: int,
                 delta_t: float,
                 model_obj: Model,
                 tlm_op_factory: TLMOpFactory,
                 tlm_adj_op_factory: TLMOpFactory | None = None,
                 B_factors: BFactors | None = None,
                 B_half_op: Callable[[ArrayLike], ArrayLike] | None = None,
                 B_half_factory: Callable[
                     [ArrayLike, Callable, int],
                     Callable[[ArrayLike], ArrayLike]] | None = None,
                 B_rank: int = 50,
                 B_probes: int = 64,
                 B_floor: float = 0.0,
                 R: ArrayLike | None = None,
                 H: ArrayLike | None = None,
                 h: Callable | None = None,
                 n_outer_loops: int = 3,
                 n_inner_loops: int = 30,
                 inner_tol: float = 1e-6,
                 steps_per_window: int = 1,
                 obs_window_indices: ArrayLike | None = None,
                 refresh_B_each_outer: bool = False,
                 lm_lambda: float = 0.0,
                 lm_relative: bool = False,
                 lm_power_iters: int = 3,
                 verbose: bool = False,
                 **kwargs
                 ):
        self.steps_per_window = steps_per_window
        self.obs_window_indices = obs_window_indices
        self.n_outer_loops = n_outer_loops
        self.n_inner_loops = n_inner_loops
        self.inner_tol = inner_tol
        self.tlm_op_factory = tlm_op_factory
        self.tlm_adj_op_factory = tlm_adj_op_factory
        self.B_factors = B_factors
        self.B_half_op = B_half_op
        self.B_half_factory = B_half_factory
        self._factory_B_half = None
        self.B_rank = int(B_rank)
        self.B_probes = int(B_probes)
        self.B_floor = float(B_floor)
        self.refresh_B_each_outer = bool(refresh_B_each_outer)
        self.lm_lambda = float(lm_lambda)
        self.lm_relative = bool(lm_relative)
        self.lm_power_iters = int(lm_power_iters)
        self.verbose = bool(verbose)

        if H is not None and not is_function_operator(H):
            H = jnp.array(H)

        super().__init__(system_dim=system_dim,
                         delta_t=delta_t,
                         model_obj=model_obj,
                         B=None, R=R, H=H, h=h)

    def _calc_default_H(self,
                        obs_loc_indices: ArrayLike
                        ) -> jax.Array:
        Hs = jnp.zeros((obs_loc_indices.shape[0], obs_loc_indices.shape[1],
                        self.system_dim),
                       dtype=jnp.float32)
        for i in range(Hs.shape[0]):
            Hs = Hs.at[i, jnp.arange(Hs.shape[1]), obs_loc_indices
                       ].set(1.0)
        return Hs

    def _calc_default_R(self,
                        obs_values: ArrayLike,
                        obs_error_sd: float
                        ) -> jax.Array:
        return jnp.identity(obs_values[0].shape[0])*(obs_error_sd**2)

    def _rollout_background(self,
                            x0_ds: XarrayDatasetLike,
                            n_steps: int
                            ) -> jax.Array:
        """Nonlinear background rollout via ``model_obj.forecast``.

        Returns the stacked trajectory as a plain JAX array of shape
        ``(n_steps, system_dim)`` (the DABench ``forecast`` contract:
        ``n_steps`` frames, last at the window boundary); the xarray
        scaffolding is unwrapped here because the inner loop is purely
        algebraic.
        """
        _, X_ds = self.model_obj.forecast(x0_ds, n_steps=n_steps)
        X_ar = X_ds.to_stacked_array('system', ['time'])
        return jnp.asarray(X_ar.data)

    def _innerloop_4d(self,
                      x_b_traj: ArrayLike,
                      v_total: ArrayLike,
                      tlm_op: Callable[[ArrayLike, ArrayLike], ArrayLike],
                      apply_B_half: Callable[[ArrayLike], ArrayLike],
                      Hs: ArrayLike,
                      obs_vals: ArrayLike,
                      obs_window_indices: ArrayLike,
                      obs_time_mask: ArrayLike,
                      R_inv_diag: ArrayLike,
                      outer_idx: int = 0,
                      aux: tuple | None = None,
                      ) -> tuple[jax.Array, dict]:
        """PCG-Lanczos inner solve in CVT v-space.

        Precomputes the innovations against the current outer's
        background trajectory, builds the matrix-free Gauss-Newton
        HVP via ``jax.jvp(jax.grad(J), ...)``, and runs PCG-Lanczos
        to refine ``v_total`` by ``delta_v``.

        ``aux = (times, inputs)`` (compiled-window path) is passed to the
        four-argument TLM ``tlm_op(x_t, dx_t, t_k, u_k)``.

        Returns:
            Tuple ``(delta_v, info)`` where ``info`` is the PCG
            diagnostics dict from :func:`pcg_lanczos_solve`.
        """
        innovations = jax.vmap(
                lambda i: obs_vals[i] - Hs[i] @ x_b_traj[obs_window_indices[i]]
                )(jnp.arange(Hs.shape[0]))

        if self.verbose:
            jax.debug.print(
                    "  [idx/outer{o}] x_b_traj.shape={s}  "
                    "obs_window_indices={owi}  obs_time_mask={otm}",
                    o=outer_idx, s=jnp.array(x_b_traj.shape),
                    owi=obs_window_indices, otm=obs_time_mask)
            innov_norms = jnp.linalg.norm(innovations, axis=-1)
            jax.debug.print(
                    "  [innov/outer{o}] per-obs ||y - H x_b||={n}",
                    o=outer_idx, n=innov_norms)

        kw_aux = {} if aux is None else dict(aux=aux)

        def J_of_dv(dv: ArrayLike) -> jax.Array:
            return quadratic_cost(
                    dv, v_total,
                    tlm_op=tlm_op,
                    x_b_traj=x_b_traj,
                    Hs=Hs,
                    innovations=innovations,
                    obs_window_indices=obs_window_indices,
                    obs_time_mask=obs_time_mask,
                    R_inv_diag=R_inv_diag,
                    apply_B_half=apply_B_half,
                    **kw_aux,
                    )

        grad_J = jax.grad(J_of_dv)
        dv0 = jnp.zeros_like(v_total)
        b_rhs = -grad_J(dv0)

        if self.verbose:
            def J_b_only(dv: ArrayLike) -> jax.Array:
                v_full = v_total + dv
                return 0.5 * jnp.sum(v_full * v_full)
            gb = jax.grad(J_b_only)(dv0)
            go = -b_rhs - gb
            jax.debug.print(
                    "  [grad/outer{o}] ||grad J_b||={gb:.3e}  "
                    "||grad J_o||={go:.3e}  ratio_o/b={r:.3e}",
                    o=outer_idx,
                    gb=jnp.linalg.norm(gb),
                    go=jnp.linalg.norm(go),
                    r=jnp.linalg.norm(go) / jnp.maximum(
                            jnp.linalg.norm(gb), 1e-30))

        # Levenberg-Marquardt damping: replace the Gauss-Newton
        # normal equation H delta_v = b with (H + lambda I) delta_v = b
        # to shrink the inner step when H is poorly conditioned or
        # when subsequent outers would relinearise in a regime where
        # the TLM is no longer valid.
        lm = self.lm_lambda

        if self.lm_relative and self.lm_lambda > 0.0:
            # The full GN Hessian in CVT v-space is (I + S), where S is the
            # observation-term curvature carrying the (sigma_bg/sigma_obs)^2
            # prefactor.  Absolute lambda*I damping is a no-op when
            # ||S|| >> 1, so scale lambda by an estimate of the spectral
            # radius of S (power iteration on S = full_hvp - I) -- making
            # lambda a dimensionless lever independent of that prefactor.
            def S_apply(p: ArrayLike) -> jax.Array:
                return jax.jvp(grad_J, (dv0,), (p,))[1] - p

            u0 = jax.random.normal(
                    jax.random.PRNGKey(int(outer_idx)),
                    dv0.shape, dtype=dv0.dtype)
            u0 = u0 / jnp.maximum(jnp.linalg.norm(u0), 1e-30)

            def _pi_step(u: ArrayLike, _: None) -> tuple[jax.Array, jax.Array]:
                w = S_apply(u)
                nrm = jnp.linalg.norm(w)
                return w / jnp.maximum(nrm, 1e-30), nrm

            _, rho_hist = jax.lax.scan(
                    _pi_step, u0, xs=None,
                    length=max(int(self.lm_power_iters), 1))
            rho_S = rho_hist[-1]
            lm = self.lm_lambda * jnp.maximum(rho_S, 1e-30)
            if self.verbose:
                jax.debug.print(
                        "  [lm/outer{o}] relative: rho(S)={r:.3e}  "
                        "lambda_eff={le:.3e}", o=outer_idx, r=rho_S, le=lm)

        def hvp_fn(p: ArrayLike) -> jax.Array:
            return jax.jvp(grad_J, (dv0,), (p,))[1] + lm * p

        delta_v, info = pcg_lanczos_solve(
                hvp_fn, b_rhs,
                max_iter=self.n_inner_loops, tol=self.inner_tol,
                verbose=self.verbose,
                verbose_tag=f"pcg/outer{int(outer_idx)}")
        if os.environ.get("DABENCH_LANCZOS_KAPPA"):
            # extreme Ritz values of the inner-loop Hessian from the CG
            # coefficients (Lanczos tridiagonal): an estimate of the
            # condition number the solver faces. Diagnostic only.
            jax.debug.callback(_print_lanczos_kappa, info["alphas"],
                               info["betas"], info["n_iter"],
                               int(outer_idx))
        if self.verbose:
            J0 = J_of_dv(dv0)
            J_star = J_of_dv(delta_v)
            jax.debug.print(
                    "  [J/outer{outer}] J(0)={J0:.3e}  "
                    "J(dv*)={Js:.3e}  ratio={r:.3e}",
                    outer=outer_idx, J0=J0, Js=J_star,
                    r=J_star / jnp.where(J0 > 0, J0, jnp.ones_like(J0)))
        return delta_v, info

    def _ensure_B_half(self,
                       x_linear: ArrayLike,
                       tlm_op: Callable[[ArrayLike, ArrayLike], ArrayLike],
                       outer_idx: int
                       ) -> Callable[[ArrayLike], ArrayLike]:
        """Resolve the ``B^(1/2)`` callable for the current outer.

        Preference order: user-supplied ``B_half_op`` >
        per-cycle ``B_half_factory`` > user-supplied ``B_factors`` >
        RSVD-built factors from :func:`extract_B_factors`.  The
        ``B_half_factory`` is invoked at outer 0 of each cycle (rebuilt
        from the current background ``x_linear``) and cached for the
        remaining outers; when ``refresh_B_each_outer`` is True the
        factory / factors are rebuilt every outer instead.
        """
        if self.B_half_op is not None:
            return self.B_half_op
        if self.B_half_factory is not None:
            if outer_idx == 0 or (self.refresh_B_each_outer and outer_idx > 0):
                self._factory_B_half = self.B_half_factory(
                        x_linear, tlm_op, outer_idx)
            return self._factory_B_half
        if (self.B_factors is None
                or (self.refresh_B_each_outer and outer_idx > 0)):
            apply_M_at_x0 = lambda dx: tlm_op(x_linear, dx)
            self.B_factors = extract_B_factors(
                    apply_M_at_x0,
                    system_dim=int(self.system_dim),
                    K=self.B_rank, E=self.B_probes,
                    seed=int(outer_idx),
                    sigma_bg=self.B_floor,
                    dtype=x_linear.dtype)
        return build_B_half(self.B_factors)

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
        """One analysis window: incremental GN with matrix-free TLM.

        Resolves ``Hs`` / ``R`` / observation alignment exactly the
        way :meth:`Var4D._cycle_obsop` does, then runs ``n_outer_loops``
        outer iterations of: (1) nonlinear rollout, (2) close TLM at
        the current linearisation, (3) ensure ``B^(1/2)``, (4) PCG-
        Lanczos inner solve in v-space, (5) accumulate ``v_total``.
        Returns the analysis ``x_a = x_b + B^(1/2) v_total`` as an
        xarray Dataset on the same coordinates as ``xb0_ds``.
        """
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
                            "Var4DOperator only supports linear H (matrix).")
            else:
                # Single H (obs_dim, system_dim); per-time mask applied to the
                # product, never materialised as (T, obs_dim, system_dim).
                Hs = make_per_time_H(
                        self.H, obs_values.shape[0], obs_loc_mask,
                        self._obs_vector.stationary_observers)

        # diagonal R stays a vector (a dense (N, N) R OOMs real networks)
        R_inv_diag = r_inv_diagonal(R if R is not None else self.R,
                                    obs_values.shape[-1], self.obs_error_sd)

        # Cache the original coords / template so we can reattach them
        # to the analysis state at the end.
        xb0_xr = (xb0_ds.to_xarray()
                  if isinstance(xb0_ds, xj.XjDataset) else xb0_ds)
        x_b0 = jnp.asarray(
                xb0_xr.to_stacked_array('system', []).data).ravel()
        v_total = jnp.zeros_like(x_b0)
        apply_B_half = None
        last_info: dict | None = None
        xb_traj0 = None                       # outer-0 background trajectory

        for outer in range(self.n_outer_loops):
            # Linearisation state for this outer:
            #   outer 0  → x_b (no v yet)
            #   outer >0 → x_b + B^(1/2) v_total
            x_l = (x_b0 if apply_B_half is None
                   else x_b0 + apply_B_half(v_total))
            x_l_ds = self._array_to_dataset_like(x_l, xb0_xr)
            x_b_traj = self._rollout_background(
                    x_l_ds, n_steps=self.steps_per_window)
            if outer == 0:
                xb_traj0 = x_b_traj

            tlm_op = self.tlm_op_factory(self.model_obj, x_b_traj[0])
            apply_B_half = self._ensure_B_half(x_b_traj[0], tlm_op, outer)

            delta_v, info = self._innerloop_4d(
                    x_b_traj, v_total, tlm_op, apply_B_half,
                    Hs, obs_values, jnp.asarray(obs_window_indices),
                    obs_time_mask, R_inv_diag,
                    outer_idx=outer)
            v_total = v_total + delta_v
            last_info = info

        # Convergence guard (behaviour-neutral): warn once per window when
        # the final inner solve did not reach inner_tol, so an under-
        # converged cycle flags itself instead of silently degrading the
        # analysis.  Emitting an aggregate k/N count across windows would
        # require changing the base cycler's scan return signature, which
        # we deliberately avoid; this prints per failing window and is
        # silent on success.
        _ni = last_info["n_iter"]
        _rel = (last_info["residual_norms"][_ni]
                / jnp.maximum(last_info["residual_norms"][0], 1e-30))
        jax.lax.cond(
                last_info["converged"],
                lambda: None,
                lambda: jax.debug.print(
                    "[warn] inner PCG not converged: n_iter={n}/{m} "
                    "rel={r:.2e} (tol={t:.0e}); if analysis is degraded "
                    "raise --n-inner / --lm-relative -- but note early "
                    "stopping can be benign regularisation at tight "
                    "sigma_obs (more iters are not always better).",
                    n=_ni, m=self.n_inner_loops, r=_rel, t=self.inner_tol),
                )

        x_a = x_b0 + apply_B_half(v_total)
        xa_ds = self._array_to_dataset_like(x_a, xb0_xr)
        if not self._return_metrics:
            return xa_ds
        dtype = x_a.dtype
        xa_traj = self._rollout_background(xa_ds, n_steps=self.steps_per_window)
        Hs_m = as_dtype(Hs, dtype)
        owi = jnp.asarray(obs_window_indices)
        Hxb = jax.vmap(lambda i: Hs_m[i] @ xb_traj0[owi[i]])(
                jnp.arange(Hs_m.shape[0]))
        Hxa = jax.vmap(lambda i: Hs_m[i] @ xa_traj[owi[i]])(
                jnp.arange(Hs_m.shape[0]))
        y = jnp.asarray(obs_values, dtype).reshape(-1)
        obs_dim = Hs_m.shape[1]
        active = (jnp.repeat(jnp.asarray(obs_time_mask, bool), obs_dim)
                  & jnp.asarray(obs_loc_mask, bool).reshape(-1))
        # R_inv_diag is per-obs-location (obs_dim,); sigma2 = 1/R_inv on active
        # entries, broadcast across the n_times obs slots to the flat obs axis.
        sigma2_loc = jnp.where(R_inv_diag > 0,
                               1.0 / jnp.where(R_inv_diag > 0, R_inv_diag,
                                               jnp.ones_like(R_inv_diag)),
                               jnp.zeros_like(R_inv_diag))
        sigma2_diag = jnp.broadcast_to(
                sigma2_loc.astype(dtype),
                (Hs_m.shape[0], obs_dim)).reshape(-1)

        # End-of-window O-A (next-cycle IC quality): analysis re-forecast to the
        # window END, scored against obs valid at the end (owi == last idx).
        end_idx = self.steps_per_window - 1
        Hxa_end = jax.vmap(lambda i: Hs_m[i] @ xa_traj[end_idx])(
                jnp.arange(Hs_m.shape[0]))
        end_active = (active & jnp.repeat(owi == end_idx, obs_dim))
        # NOTE: obs-space SPREAD overrides are omitted here (=> NaN, schema
        # uniform with Var4DBackprop).  This is a MATRIX-FREE cycler: B and the
        # window TLM are applied as operators (apply_B_half / tlm_op), never
        # materialised, precisely to avoid O(system_dim) dense cost.  Deriving
        # obs_space_spread_background / _analysis_end (as Var4D does) would
        # require densifying B_half and M via system_dim matvecs per cycle,
        # defeating the operator design; the O-F/O-A innovations that training
        # and forensics consume do NOT depend on the spreads.
        metrics = dac_utils._obs_space_metrics(
                y, Hxb.reshape(-1).astype(dtype), Hxa.reshape(-1).astype(dtype),
                active, sigma2_diag, ens_obs=None,
                return_per_obs=(self._metrics_mode == "debug"), dtype=dtype,
                Hxa_end_mean=Hxa_end.reshape(-1).astype(dtype),
                end_active_mask=end_active)
        return xa_ds, metrics

    # ------------------------------------------------------------------
    # Compiled-window path (``cycle(model_inputs=..., obs_inputs=...,
    # compile_once=True)``): the window is one jax.jit-ed function of the
    # per-cycle data, built once per cycler instance and configuration.
    # ------------------------------------------------------------------
    _supports_inputs: bool = True

    def _model_step_inputs(self) -> Callable:
        """The model step as ``F(x, t, u)`` on a flat state, detected once.

        ``model_obj.step_fn()`` may return ``F(x)`` (time-independent model,
        wrapped) or ``F(x, t, u)`` (``t`` the model time of the step, ``u``
        its row of ``model_inputs``, None when there are none).
        """
        F3 = getattr(self, "_step3", None)
        if F3 is None:
            if not callable(getattr(self.model_obj, "step_fn", None)):
                raise NotImplementedError(
                    "the compiled-window path needs model_obj.step_fn() "
                    "returning F(x) or F(x, t, u) on a flat state")
            F = self.model_obj.step_fn()
            F3 = F if _n_positional(F) >= 3 else (lambda x, t, u: F(x))
            self._step3 = F3
        return F3

    def _tlm_op_inputs(self, x_linear: ArrayLike) -> Callable:
        """Four-argument TLM ``tlm_op(x_t, dx_t, t_k, u_k)``.

        A user ``tlm_op_factory`` whose op takes four arguments is used as
        is; one whose op takes two (time-independent) is wrapped; with no
        factory the TLM is ``jax.jvp`` of the model step at ``(t_k, u_k)``.
        """
        if self.tlm_op_factory is not None:
            op = self.tlm_op_factory(self.model_obj, x_linear)
            if _n_positional(op) >= 4:
                return op
            return lambda x_t, dx_t, t_k, u_k: op(x_t, dx_t)
        F = self._model_step_inputs()
        return lambda x_t, dx_t, t_k, u_k: jax.jvp(
                lambda x: F(x, t_k, u_k), (x_t,), (dx_t,))[1]

    def _check_B_compiled(self) -> None:
        if (self.B_half_op is None and self.B_half_factory is None
                and self.B_factors is None):
            raise NotImplementedError(
                "the compiled-window path needs B_half_op, B_half_factory or "
                "B_factors (the RSVD-built B is stored on the instance inside "
                "the trace and cannot be reused by later calls)")

    def _resolve_Hs_inputs(self, obs_values, obs_loc_indices, obs_loc_mask,
                           obs_inputs):
        """``Hs`` as in :meth:`_cycle_obsop`, with ``obs_inputs`` bound per
        time when ``H`` takes them."""
        if self.H is None:
            if self.h is not None:
                raise ValueError(
                    "Var4DOperator only supports linear H (matrix).")
            if obs_inputs is not None:
                raise ValueError(
                    "obs_inputs need a LinearObsOperator H with "
                    "takes_inputs=True")
            H = self._calc_default_H(obs_loc_indices)
            return jax.lax.cond(
                    self._obs_vector.stationary_observers,
                    lambda: H,
                    lambda: (obs_loc_mask[:, :, jnp.newaxis] * H))
        return make_per_time_H(
                self.H, obs_values.shape[0], obs_loc_mask,
                self._obs_vector.stationary_observers, obs_inputs=obs_inputs)

    def _window_inputs(self, x_b0, obs_vals, obs_loc_indices, obs_loc_mask,
                       obs_time_mask, obs_times, obs_error_sd, cur_time, u,
                       o):
        """One window on flat arrays: ``(x_a, trajectory (S, D))``, the
        trajectory being the forecast from ``x_a`` over the window (its last
        frame is the next background).  Strong constraint; see
        :class:`Var4DOperatorWC` for the forced version."""
        S = int(self.steps_per_window)
        times = cur_time + jnp.arange(S - 1) * self.delta_t
        owi = jnp.argmin(jnp.abs(obs_times[:, None]
                                 - (cur_time + self._model_timesteps)[None]),
                         axis=1)
        Hs = self._resolve_Hs_inputs(obs_vals, obs_loc_indices,
                                     obs_loc_mask, o)
        R_inv_diag = r_inv_diagonal(self.R, obs_vals.shape[-1], obs_error_sd)
        F = self._model_step_inputs()
        aux = (times, u)
        u0 = jax.tree_util.tree_map(lambda a: a[0], u)
        v_total = jnp.zeros_like(x_b0)
        apply_B_half = None
        for outer in range(self.n_outer_loops):
            x_l = (x_b0 if apply_B_half is None
                   else x_b0 + apply_B_half(v_total))
            x_b_traj = model_rollout(F, x_l, times, u)
            tlm_op = self._tlm_op_inputs(x_b_traj[0])
            apply_B_half = self._ensure_B_half(
                    x_b_traj[0],
                    lambda x, dx: tlm_op(x, dx, times[0], u0), outer)
            delta_v, info = self._innerloop_4d(
                    x_b_traj, v_total, tlm_op, apply_B_half, Hs, obs_vals,
                    owi, obs_time_mask, R_inv_diag, outer_idx=outer, aux=aux)
            v_total = v_total + delta_v
        x_a = x_b0 + apply_B_half(v_total)
        return x_a, model_rollout(F, x_a, times, u)

    def _compiled_window(self) -> Callable:
        """The jitted window, rebuilt only if the window configuration
        changes (the per-cycle data are its arguments)."""
        key = (int(self.steps_per_window), float(self.delta_t),
               bool(self._obs_vector.stationary_observers))
        if getattr(self, "_window_key", None) != key:
            self._window_key = key
            self._window_jit = jax.jit(self._window_inputs)
        return self._window_jit

    def reset_compiled(self) -> None:
        """Forget the cached model step and compiled window (call after
        changing the model or the cycler's operators in place)."""
        self._step3 = None
        self._window_key = None
        self._window_jit = None

    def _cycle_with_inputs(self, input_state, start_time, obs_vector,
                           n_cycles, obs_error_sd, analysis_window,
                           analysis_time_in_window, return_forecast,
                           return_final_state, return_background,
                           model_inputs, obs_inputs, obs_times_per_window):
        """``cycle()`` on the compiled-window path: host-side gathers of each
        cycle's observations and inputs (fixed shapes), one jitted window per
        cycle, the same return structure as the original path."""
        self._check_B_compiled()
        self._return_metrics = False
        self.metrics = None
        input_state, all_filtered_padded = self._prepare_cycle(
            input_state, start_time, obs_vector, obs_error_sd, n_cycles,
            analysis_window, analysis_time_in_window)
        S = int(self.steps_per_window)
        n_win_steps = S - 1
        template = input_state.drop_vars('_cur_time')
        st = template.to_stacked_array('system', [])
        x_b = jnp.asarray(st.data).ravel()

        obs_all = np.asarray(self._obs_vector[self._observed_vars]
                             .to_stacked_array('system', ['time']).data)
        times_all = np.asarray(self._obs_vector.time.data, dtype=float)
        loc_all = np.asarray(self._obs_vector.system_index.data)
        mask_all = np.asarray(self._obs_loc_masks)
        fp = np.asarray(all_filtered_padded)
        n_win = fp.shape[1]
        if obs_times_per_window is not None:
            if obs_times_per_window < n_win:
                raise ValueError(
                    f"obs_times_per_window={obs_times_per_window} < "
                    f"{n_win} observation times in a window")
            fp = np.concatenate(
                [fp, np.zeros((fp.shape[0], obs_times_per_window - n_win),
                              fp.dtype)], axis=1)
            n_win = obs_times_per_window
        if model_inputs is not None:
            need = int(n_cycles) * n_win_steps
            for leaf in jax.tree_util.tree_leaves(model_inputs):
                if leaf.shape[0] < need:
                    raise ValueError(
                        f"model_inputs leaves need >= {need} rows "
                        f"(n_cycles * (steps_per_window - 1)), got "
                        f"{leaf.shape[0]}")

        window = self._compiled_window()
        sd = jnp.asarray(self.obs_error_sd, dtype=x_b.dtype)
        trajs = []
        for c in range(int(n_cycles)):
            fi = fp[c] - 1          # -1 on padding (masked), as the scan path
            u = (None if model_inputs is None else jax.tree_util.tree_map(
                    lambda a: jnp.asarray(
                        np.asarray(a)[c * n_win_steps:(c + 1) * n_win_steps]),
                    model_inputs))
            o = (None if obs_inputs is None else jax.tree_util.tree_map(
                    lambda a: jnp.asarray(np.asarray(a)[fi]), obs_inputs))
            cur_time = jnp.asarray(float(start_time) + c * float(analysis_window),
                                   dtype=x_b.dtype)
            x_a, traj = window(
                    x_b, jnp.asarray(obs_all[fi]),
                    jnp.asarray(loc_all[:, fi].reshape(n_win, -1)),
                    jnp.asarray(mask_all[:, fi].astype(bool).reshape(n_win, -1)),
                    jnp.asarray(fp[c] > 0),
                    jnp.asarray(times_all[fi], dtype=x_b.dtype), sd, cur_time,
                    u, o)
            trajs.append(traj)
            x_b = traj[-1]

        def to_ds(traj):
            return xr.DataArray(
                np.asarray(traj), dims=('time', 'system'),
                coords={'system': st['system']}).to_unstacked_dataset('system')

        frames = [to_ds(t) for t in trajs]
        all_vals_ds = xr.Dataset(
            {var: (('cycle',) + tuple(frames[0][var].dims),
                   np.stack([f[var].data for f in frames]))
             for var in frames[0].data_vars}
        ).rename_dims({'time': 'cycle_timestep'})
        analysis_ds = (all_vals_ds.drop_isel(cycle_timestep=-1)
                       if return_forecast
                       else all_vals_ds.isel(cycle_timestep=0))
        result = [analysis_ds]
        if return_final_state:
            result.append(frames[-1].isel(time=-1, drop=True))
        if return_background:
            result.append(all_vals_ds.isel(cycle_timestep=-1))
        if len(result) == 1:
            return result[0]
        return tuple(result)

    @staticmethod
    def _array_to_dataset_like(arr: ArrayLike,
                               template: xr.Dataset
                               ) -> xr.Dataset:
        """Reshape a flat ``(system_dim,)`` array back to ``template``'s
        variable layout.

        Mirrors the inverse of ``template.to_stacked_array('system', [])``;
        used to re-wrap analysis states as xarray for downstream
        consumers (forecast, metrics).
        """
        stacked = template.to_stacked_array('system', [])
        stacked = stacked.copy(data=jnp.asarray(arr))
        return stacked.to_unstacked_dataset('system').assign_attrs(
                template.attrs)


def _n_positional(fn: Callable) -> int:
    """Number of positional parameters of ``fn`` (a large number for
    ``*args``); used to tell ``F(x)`` from ``F(x, t, u)`` and two- from
    four-argument TLM callables."""
    try:
        params = inspect.signature(fn).parameters.values()
    except (TypeError, ValueError):
        return 1
    n = 0
    for p in params:
        if p.kind == p.VAR_POSITIONAL:
            return 99
        if p.kind in (p.POSITIONAL_ONLY, p.POSITIONAL_OR_KEYWORD):
            n += 1
    return n
