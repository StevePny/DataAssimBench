"""Matrix-free backpropagation 4D-Var with a limited-memory Gauss-Newton preconditioner.

:class:`Var4DBackprop` preconditions its gradient steps with the dense inverse of
``B^-1 + H^T R^-1 H`` (a ``system_dim^2`` matrix), which cannot be formed for a
large state. This cycler keeps the backpropagation idea (the gradient of the
full nonlinear 4D-Var cost, back-propagated through the nonlinear model window
with ``jax.grad``; no linearised inner loop) and replaces the dense inverse by a
matrix-free one, on the compiled-window path of :class:`Var4DOperator`
(``cycle(..., model_inputs=..., compile_once=True)``):

* control space: ``x_0 = x_b + B^(1/2) v`` (``B_half_op`` / ``B_factors``), so
  the background term is ``1/2 |v|^2`` and the Gauss-Newton Hessian is
  ``A = I + B^(T/2) L^T R^-1 L B^(1/2)`` (``L`` the tangent-linear window
  followed by ``H``), with every eigenvalue >= 1;
* once per window, ``n_lanczos`` Lanczos steps on ``A`` at the background
  trajectory (each one Hessian-vector product: a tangent-linear pass and an
  adjoint pass), started from the first gradient, give its leading Ritz pairs
  ``(lam_i, u_i)``;
* the limited-memory inverse ``P = I + sum_i (1 / lam_i - 1) u_i u_i^T`` is
  exact on the leading eigenvectors and the identity elsewhere (where ``A`` is
  close to ``I``); applying it costs ``n_lanczos`` dot products;
* ``num_iters`` optimizer steps (``optax``: ``sgd`` or ``adam``, learning rate
  and exponential decay as in :class:`Var4DBackprop`) on ``P grad J(v)``, each
  one nonlinear forward pass and its backward pass.

With a quadratic cost, ``lr = 1`` and ``n_lanczos`` at least the number of
significant eigenvalues, the first step is the Gauss-Newton step; on a
nonlinear window the later steps use the true gradient, so the analysis moves
along the nonlinear cost without re-linearising, and converges to the minimum
the incremental 4D-Var finds (tested). The learning rate, its decay and the
optimizer are hyperparameters to calibrate (e.g. with Optuna).

Strong constraint. Requires ``B_half_op``, ``B_half_factory`` or ``B_factors``
and the compiled-window path.
"""
from typing import Callable

import jax
import jax.numpy as jnp
import optax

from dabench.dacycler._per_time_h import r_inv_diagonal
from dabench.dacycler._var4d_operator import Var4DOperator
from dabench.dacycler._var4d_weak_utils import model_rollout

OPTIMIZERS = {"sgd": optax.sgd, "adam": optax.adam}


def lanczos_eigs(matvec: Callable, q0: jax.Array, k: int):
    """``k`` Lanczos steps (full reorthogonalisation) of the symmetric operator
    ``matvec`` from ``q0``: the Ritz values ``(k,)`` and vectors ``(k, n)``."""
    n = q0.shape[0]
    q = q0 / jnp.linalg.norm(q0)
    Q = jnp.zeros((k, n), q0.dtype)
    alpha = jnp.zeros(k, q0.dtype)
    beta = jnp.zeros(k, q0.dtype)

    def body(i, carry):
        Q, alpha, beta, q, q_prev, b_prev = carry
        Q = Q.at[i].set(q)
        w = matvec(q) - b_prev * q_prev
        a = jnp.vdot(q, w)
        w = w - a * q
        w = w - Q.T @ (Q @ w)          # full reorthogonalisation (rows beyond i are zero)
        b = jnp.linalg.norm(w)
        q_next = jnp.where(b > 0, w / jnp.where(b > 0, b, 1.0), jnp.zeros_like(w))
        return (Q, alpha.at[i].set(a), beta.at[i].set(b), q_next, q, b)

    Q, alpha, beta, *_ = jax.lax.fori_loop(
        0, k, body, (Q, alpha, beta, q, jnp.zeros_like(q), jnp.asarray(0.0, q0.dtype)))
    T = jnp.diag(alpha) + jnp.diag(beta[:-1], 1) + jnp.diag(beta[:-1], -1)
    lam, S = jnp.linalg.eigh(T)
    return lam, S.T @ Q


def backtrack(cost: Callable, x: jax.Array, upd: jax.Array, J: jax.Array,
              g: jax.Array, max_halvings: int, c: float = 1e-4):
    """Armijo backtracking along the optimizer update ``upd`` from ``x``:
    the step ``t upd`` with ``t = 1, 1/2, ...`` (at most ``max_halvings``
    halvings) whose cost is finite and below ``J + c t g.upd``; ``0`` (no
    move) if none is. Returns the new point."""
    slope = jnp.vdot(g, upd)

    def ok(t):
        Jt = cost(x + t * upd)
        return jnp.isfinite(Jt) & (Jt <= J + c * t * jnp.minimum(slope, 0.0))

    def cond(carry):
        t, k, good = carry
        return (~good) & (k < max_halvings)

    def body(carry):
        t, k, _ = carry
        t = 0.5 * t
        return t, k + 1, ok(t)

    t0 = jnp.asarray(1.0, x.dtype)
    t, _, good = jax.lax.while_loop(cond, body, (t0, 0, ok(t0)))
    return x + jnp.where(good, t, 0.0) * upd


def lm_inverse(lam: jax.Array, U: jax.Array) -> Callable:
    """``P = I + sum_i (1 / lam_i - 1) u_i u_i^T`` (eigenvalues below 1 are
    clipped to 1: the Gauss-Newton Hessian in control space has none)."""
    c = 1.0 / jnp.maximum(lam, 1.0) - 1.0

    def apply(g):
        return g + U.T @ (c * (U @ g))

    return apply


class Var4DBackpropMF(Var4DOperator):
    """Matrix-free backpropagation 4D-Var (see the module docstring).

    Args:
        num_iters: Optimizer steps per window. Default 10.
        learning_rate: Optimizer learning rate. Default 1.0 (the
            Gauss-Newton step length in the preconditioned space).
        lr_decay: Exponential decay of the learning rate per step (1: none).
            Default 0.9: a constant rate of 1 oscillates about the minimum of
            a nonlinear window (toy test: stuck at 0.065 from the incremental
            analysis), 0.5 or a decay converges to it.
        optimizer: ``"sgd"`` (default) or ``"adam"``.
        n_lanczos: Lanczos steps (Ritz pairs) of the preconditioner per
            window; 0 switches it off (plain gradient in control space).
            Default 20.
        line_search: Maximum number of step halvings of an Armijo
            backtracking line search on each optimizer step (the step is
            halved until the nonlinear cost is finite and decreases enough;
            no move if it never does). 0 (default): the plain optimizer step.
        **kwargs: As :class:`Var4DOperator` (``B_half_op`` etc.;
            ``n_outer_loops`` / ``n_inner_loops`` are not used).
    """

    def __init__(self, *args, num_iters: int = 10, learning_rate: float = 1.0,
                 lr_decay: float = 0.9, optimizer: str = "sgd",
                 n_lanczos: int = 20, line_search: int = 0, **kwargs):
        super().__init__(*args, **kwargs)
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
        S = int(self.steps_per_window)
        times = cur_time + jnp.arange(S - 1) * self.delta_t
        owi = jnp.argmin(jnp.abs(obs_times[:, None]
                                 - (cur_time + self._model_timesteps)[None]),
                         axis=1)
        Hs = self._resolve_Hs_inputs(obs_vals, obs_loc_indices,
                                     obs_loc_mask, o)
        R_inv = r_inv_diagonal(self.R, obs_vals.shape[-1], obs_error_sd)
        F = self._model_step_inputs()
        u0 = jax.tree_util.tree_map(lambda a: a[0], u)
        n_obs = obs_vals.shape[0]
        idx = jnp.arange(n_obs)
        valid = (jnp.asarray(obs_time_mask)[:, None]
                 & jnp.isfinite(obs_vals)
                 & (jnp.broadcast_to(R_inv, obs_vals.shape) > 0))
        y = jnp.where(valid, obs_vals, 0.0)
        w = jnp.where(valid, jnp.broadcast_to(R_inv, obs_vals.shape), 0.0)

        xb_traj = model_rollout(F, x_b0, times, u)
        tlm_op = self._tlm_op_inputs(xb_traj[0])
        apply_B_half = self._ensure_B_half(
                xb_traj[0], lambda x, dx: tlm_op(x, dx, times[0], u0), 0)

        def hx(traj):
            return jax.vmap(lambda i: Hs[i] @ traj[owi[i]])(idx)

        def cost(v):
            traj = model_rollout(F, x_b0 + apply_B_half(v), times, u)
            r = jnp.where(valid, hx(traj) - y, 0.0)
            return 0.5 * jnp.sum(v * v) + 0.5 * jnp.sum(w * r * r)

        v = jnp.zeros_like(x_b0)
        grad_J = jax.grad(cost)
        g0 = grad_J(v)
        if self.n_lanczos > 0:
            def G(vv):  # observation-space image of the control, nonlinear
                traj = model_rollout(F, x_b0 + apply_B_half(vv), times, u)
                return hx(traj)

            def gn_hvp(p):  # (I + J^T W J) p at the background, J = dG/dv
                Jp = jax.jvp(G, (v,), (p,))[1]
                _, vjp = jax.vjp(G, v)
                return p + vjp(jnp.where(valid, w * Jp, 0.0))[0]

            lam, U = lanczos_eigs(gn_hvp, g0, self.n_lanczos)
            P = lm_inverse(lam, U)
        else:
            lam = jnp.zeros(0, x_b0.dtype)
            P = lambda g: g  # noqa: E731

        sched = optax.exponential_decay(self.learning_rate, 1, self.lr_decay)
        opt = OPTIMIZERS[self.optimizer](sched)
        state = opt.init(v)

        def step(carry, _):
            v, state = carry
            J, g = jax.value_and_grad(cost)(v)
            upd, state = opt.update(P(g), state)
            if self.line_search:
                return (backtrack(cost, v, upd, J, g, self.line_search), state), J
            return (optax.apply_updates(v, upd), state), J

        (v, _), Js = jax.lax.scan(step, (v, state), None, length=self.num_iters)
        x_a = x_b0 + apply_B_half(v)
        if self.verbose:
            jax.debug.print("[backprop-mf] cost {j0:.6e} -> {j1:.6e} in {n} steps; Ritz values {lo:.3g}..{hi:.3g}",
                            j0=Js[0], j1=Js[-1], n=self.num_iters, lo=jnp.min(lam, initial=1.0), hi=jnp.max(lam, initial=1.0))
        return x_a, model_rollout(F, x_a, times, u)
