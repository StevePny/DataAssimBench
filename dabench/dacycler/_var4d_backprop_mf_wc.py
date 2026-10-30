"""Weak-constraint matrix-free backpropagation 4D-Var.

The weak-constraint counterpart of :class:`Var4DBackpropMF`, on the forcing
formulation of :class:`Var4DOperatorWC` (compiled-window path):

* controls ``z = (v, chi)``: ``x_0 = x_b + B^(1/2) v`` and the model error
  ``eta = Q^(1/2) chi`` added to every model step, ``x_{k+1} = M(x_k) + eta_k``
  (one ``chi`` for the window with ``eta_mode="constant"``, one per step with
  ``"per_step"``);
* cost ``J(z) = 1/2 |v|^2 + 1/2 |chi|^2 + 1/2 sum (H x - y)^T R^-1 (H x - y)``
  over the forced nonlinear window, its gradient back-propagated through the
  window with ``jax.grad`` (no linearised inner loop);
* preconditioner: ``n_lanczos`` Lanczos pairs of the Gauss-Newton Hessian
  ``I + J^T R^-1 J`` in ``z`` at the background (``J`` = d(H x)/dz), as the
  strong-constraint class;
* ``num_iters`` optimizer steps (``sgd`` / ``adam``, learning rate with
  exponential decay) on the preconditioned gradient.

The returned trajectory is the window forced with the fitted ``eta`` when
``forecast_with_eta`` (as :class:`Var4DOperatorWC`), so callers that recover
``eta_k = x_{k+1} - M(x_k)`` from it get the analysed forcing. Without
``Q_half`` the class falls back to :class:`Var4DBackpropMF` (strong constraint).
"""
import jax
import jax.numpy as jnp
import optax

from dabench.dacycler._per_time_h import r_inv_diagonal
from dabench.dacycler._var4d_backprop_mf import OPTIMIZERS, Var4DBackpropMF, backtrack, lanczos_eigs, lm_inverse
from dabench.dacycler._var4d_operator_wc import Var4DOperatorWC
from dabench.dacycler._var4d_weak_utils import expand_eta, forced_rollout, model_rollout, n_eta


class Var4DBackpropMFWC(Var4DOperatorWC):
    """Weak-constraint matrix-free backpropagation 4D-Var (see the module docstring).

    Args:
        num_iters, learning_rate, lr_decay, optimizer, n_lanczos, line_search, precond: as
            :class:`Var4DBackpropMF`.
        **kwargs: As :class:`Var4DOperatorWC` (``Q_half``, ``eta_mode``,
            ``B_half_op`` etc.; ``n_outer_loops`` / ``n_inner_loops`` are not
            used).
    """

    def __init__(self, *args, num_iters: int = 10, learning_rate: float = 1.0,
                 lr_decay: float = 0.9, optimizer: str = "sgd",
                 n_lanczos: int = 20, line_search: int = 0, precond: str = "gn", **kwargs):
        super().__init__(*args, **kwargs)
        if precond not in ("gn", "obs"):
            raise ValueError(f"precond {precond!r}: 'gn' or 'obs'")
        self.precond = precond
        if optimizer not in OPTIMIZERS:
            raise ValueError(f"optimizer {optimizer!r}: {sorted(OPTIMIZERS)}")
        self.num_iters = int(num_iters)
        self.learning_rate = float(learning_rate)
        self.lr_decay = float(lr_decay)
        self.optimizer = optimizer
        self.n_lanczos = int(n_lanczos)
        self.line_search = int(line_search)

    def _window_inputs(self, x_b0, obs_vals, obs_loc_indices, obs_loc_mask,
                       obs_time_mask, obs_times, obs_error_sd, cur_time, u,
                       o):
        if self._apply_Q_half is None:
            return Var4DBackpropMF._window_inputs(
                    self, x_b0, obs_vals, obs_loc_indices, obs_loc_mask,
                    obs_time_mask, obs_times, obs_error_sd, cur_time, u, o)
        S = int(self.steps_per_window)
        T = S - 1
        times = cur_time + jnp.arange(T) * self.delta_t
        owi = jnp.argmin(jnp.abs(obs_times[:, None]
                                 - (cur_time + self._model_timesteps)[None]),
                         axis=1)
        Hs = self._resolve_Hs_inputs(obs_vals, obs_loc_indices,
                                     obs_loc_mask, o)
        R_inv = r_inv_diagonal(self.R, obs_vals.shape[-1], obs_error_sd)
        F = self._model_step_inputs()
        u0 = jax.tree_util.tree_map(lambda a: a[0], u)
        D = x_b0.shape[0]
        ne = n_eta(self.eta_mode, T + 1)
        idx = jnp.arange(obs_vals.shape[0])
        valid = (jnp.asarray(obs_time_mask)[:, None]
                 & jnp.isfinite(obs_vals)
                 & (jnp.broadcast_to(R_inv, obs_vals.shape) > 0))
        y = jnp.where(valid, obs_vals, 0.0)
        w = jnp.where(valid, jnp.broadcast_to(R_inv, obs_vals.shape), 0.0)

        xb_traj = model_rollout(F, x_b0, times, u)
        tlm_op = self._tlm_op_inputs(xb_traj[0])
        apply_B_half = self._ensure_B_half(
                xb_traj[0], lambda x, dx: tlm_op(x, dx, times[0], u0), 0)

        def split(z):
            return z[:D], z[D:].reshape(ne, D)

        def eta_seq(chi):
            return expand_eta(jax.vmap(self._apply_Q_half)(chi), T)

        def window(z):
            v, chi = split(z)
            return forced_rollout(F, x_b0 + apply_B_half(v), eta_seq(chi),
                                  times=times, inputs=u)

        def hx(traj):
            return jax.vmap(lambda i: Hs[i] @ traj[owi[i]])(idx)

        def cost(z):
            r = jnp.where(valid, hx(window(z)) - y, 0.0)
            return 0.5 * jnp.sum(z * z) + 0.5 * jnp.sum(w * r * r)

        z = jnp.zeros(D * (1 + ne), x_b0.dtype)
        g0 = jax.grad(cost)(z)
        if self.n_lanczos > 0:
            def G(zz):  # observation-space image of the control, nonlinear (precond "obs": linear, no model: x_k = x_0 + k eta)
                if self.precond == "obs":
                    v_, chi_ = split(zz)
                    et = eta_seq(chi_)
                    acc = jnp.concatenate([jnp.zeros((1, D), x_b0.dtype), jnp.cumsum(et, axis=0)], axis=0)
                    return hx(x_b0 + apply_B_half(v_) + acc)
                return hx(window(zz))

            def gn_hvp(p):  # (I + J^T W J) p at the background, J = dG/dz
                Jp = jax.jvp(G, (z,), (p,))[1]
                _, vjp = jax.vjp(G, z)
                return p + vjp(jnp.where(valid, w * Jp, 0.0))[0]

            lam, U = lanczos_eigs(gn_hvp, g0, self.n_lanczos)
            P = lm_inverse(lam, U)
        else:
            lam = jnp.zeros(0, x_b0.dtype)
            P = lambda g: g  # noqa: E731

        sched = optax.exponential_decay(self.learning_rate, 1, self.lr_decay)
        opt = OPTIMIZERS[self.optimizer](sched)
        state = opt.init(z)

        def step(carry, _):
            z, state = carry
            J, g = jax.value_and_grad(cost)(z)
            upd, state = opt.update(P(g), state)
            if self.line_search:
                return (backtrack(cost, z, upd, J, g, self.line_search), state), J
            return (optax.apply_updates(z, upd), state), J

        (z, _), Js = jax.lax.scan(step, (z, state), None, length=self.num_iters)
        v, chi = split(z)
        x_a = x_b0 + apply_B_half(v)
        if self.verbose:
            jax.debug.print("[backprop-mf-wc] cost {j0:.6e} -> {j1:.6e} in {n} steps; Ritz values {lo:.3g}..{hi:.3g}",
                            j0=Js[0], j1=Js[-1], n=self.num_iters, lo=jnp.min(lam, initial=1.0), hi=jnp.max(lam, initial=1.0))
        if self.forecast_with_eta:
            return x_a, forced_rollout(F, x_a, eta_seq(chi), times=times, inputs=u)
        return x_a, model_rollout(F, x_a, times, u)
