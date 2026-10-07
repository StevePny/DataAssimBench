"""Utils for weak-constraint (model-error forcing) 4D-Var.

Forcing formulation (ECMWF-style).  The window trajectory carries an
additive model-error forcing ``eta_k``:

    x_{k+1} = M(x_k) + eta_k,     k = 0, ..., T - 1

and the cost gains a model-error term

    J = 1/2 dx0^T B^-1 dx0 + 1/2 sum_k eta_k^T Q^-1 eta_k + J_o.

Two placements of ``eta`` are supported (``eta_mode``):

  * ``"constant"``: a single forcing ``eta`` shared by every model step
    of the window (bias-like; ECMWF practice).  Control shape ``(1, D)``.
  * ``"per_step"``: an independent ``eta_k`` per model step.  Control
    shape ``(T, D)`` with ``T = steps_per_window - 1``.

``eta`` is preconditioned with ``Q^(1/2)`` exactly as ``dx0`` is with
``B^(1/2)``: ``eta_k = Q^(1/2) chi_k`` so ``J_q = 1/2 sum_k |chi_k|^2``.
"""

from typing import Callable

import jax
import jax.numpy as jnp
import numpy as np

from dabench.dacycler._var4d_operator_utils import BFactors, build_B_half


# For typing
ArrayLike = np.ndarray | jax.Array
QHalfLike = (float | ArrayLike | BFactors
             | Callable[[ArrayLike], ArrayLike] | None)

ETA_MODES = ("constant", "per_step")


def check_eta_mode(eta_mode: str) -> str:
    """Validate ``eta_mode`` (``"constant"`` or ``"per_step"``)."""
    if eta_mode not in ETA_MODES:
        raise ValueError(
                f"eta_mode must be one of {ETA_MODES}; got {eta_mode!r}.")
    return eta_mode


def n_eta(eta_mode: str, n_frames: int) -> int:
    """Number of independent forcing vectors for a window of
    ``n_frames`` trajectory frames (``n_frames - 1`` model steps)."""
    return 1 if eta_mode == "constant" else max(int(n_frames) - 1, 1)


def build_Q_half(Q_half: QHalfLike,
                 system_dim: int
                 ) -> Callable[[ArrayLike], ArrayLike] | None:
    """Resolve a user-supplied ``Q^(1/2)`` into a callable.

    Accepted forms:
      * ``None``: no model-error term (strong constraint) -> ``None``.
      * scalar ``sigma_q``: ``Q^(1/2) = sigma_q I``.
      * 1-D array ``(system_dim,)``: diagonal ``Q^(1/2) = diag(q)``.
      * 2-D array ``(system_dim, system_dim)``: square-root factor ``L``
        with ``Q = L L^T`` (applied as ``L @ chi``).
      * :class:`BFactors`: closed with :func:`build_B_half`.
      * callable ``chi -> eta``: used as is (must be linear).
    """
    if Q_half is None:
        return None
    if isinstance(Q_half, BFactors):
        return build_B_half(Q_half)
    if callable(Q_half):
        return Q_half
    q = jnp.asarray(Q_half)
    if q.ndim == 0:
        return lambda chi: q * chi
    if q.ndim == 1:
        if q.shape[0] != system_dim:
            raise ValueError(
                    f"diagonal Q_half has length {q.shape[0]}, "
                    f"expected system_dim={system_dim}.")
        return lambda chi: q * chi
    if q.ndim == 2:
        if q.shape != (system_dim, system_dim):
            raise ValueError(
                    f"matrix Q_half must be ({system_dim}, {system_dim}); "
                    f"got {tuple(q.shape)}.")
        return lambda chi: q @ chi
    raise ValueError(f"Unsupported Q_half with ndim={q.ndim}.")


def expand_eta(eta: ArrayLike, n_steps: int) -> jax.Array:
    """Broadcast an ``(n_eta, D)`` forcing to one row per model step."""
    eta = jnp.asarray(eta)
    if eta.shape[0] == n_steps:
        return eta
    return jnp.broadcast_to(eta[:1], (n_steps,) + eta.shape[1:])


def forced_rollout(step_fn: Callable[[ArrayLike], ArrayLike],
                   x0: ArrayLike,
                   eta_seq: ArrayLike,
                   ) -> jax.Array:
    """Nonlinear forced trajectory ``x_{k+1} = step_fn(x_k) + eta_k``.

    Args:
        step_fn: One model step on a flat ``(D,)`` state.
        x0: Initial state ``(D,)``.
        eta_seq: Per-step forcing ``(T, D)``.

    Returns:
        Trajectory ``(T + 1, D)`` with ``traj[0] = x0``.
    """
    def body(x, eta_k):
        x_next = step_fn(x) + eta_k
        return x_next, x_next

    _, tail = jax.lax.scan(body, x0, eta_seq)
    return jnp.concatenate([x0[None, :], tail], axis=0)


def window_tlm_rollout_forced(
        tlm_op: Callable[[ArrayLike, ArrayLike], ArrayLike],
        x_traj: ArrayLike,
        dx0: ArrayLike,
        deta_seq: ArrayLike,
        ) -> jax.Array:
    """Forced TLM rollout ``dx_{k+1} = M_k dx_k + deta_k``.

    Same conventions as :func:`window_tlm_rollout`; ``deta_seq`` has
    shape ``(T, D)`` with ``T = x_traj.shape[0] - 1``.
    """
    T = x_traj.shape[0] - 1

    def step(dx_t, inp):
        x_t, deta_t = inp
        dx_next = tlm_op(x_t, dx_t) + deta_t
        return dx_next, dx_next

    _, dx_tail = jax.lax.scan(step, dx0, (x_traj[:T], deta_seq))
    return jnp.concatenate([dx0[None, :], dx_tail], axis=0)


def quadratic_cost_wc(
        delta_v: ArrayLike,
        delta_chi: ArrayLike,
        v_total: ArrayLike,
        chi_total: ArrayLike,
        tlm_op: Callable[[ArrayLike, ArrayLike], ArrayLike],
        x_l_traj: ArrayLike,
        Hs: ArrayLike,
        innovations: ArrayLike,
        obs_window_indices: ArrayLike,
        obs_time_mask: ArrayLike,
        R_inv_diag: ArrayLike,
        apply_B_half: Callable[[ArrayLike], ArrayLike],
        apply_Q_half: Callable[[ArrayLike], ArrayLike],
        ) -> jax.Array:
    """Incremental weak-constraint 4D-Var cost in CVT space.

    ``J = 1/2 |v_total + delta_v|^2 + 1/2 |chi_total + delta_chi|^2
          + 1/2 sum_i |H_i dx_{j(i)} - d_i|^2_{R^-1}``

    with ``dx_0 = B^(1/2) delta_v``, ``deta_k = Q^(1/2) delta_chi_k``
    (broadcast over steps when ``delta_chi`` has a single row), the
    forced TLM recursion of :func:`window_tlm_rollout_forced` along the
    current outer's forced linearisation trajectory ``x_l_traj``, and
    innovations ``d_i = y_i - H_i x_l_traj_{j(i)}`` against it.

    Note: the increment ``dx`` is taken relative to the linearisation
    trajectory (only the *inner* increment is propagated), so this is
    the exact Gauss-Newton model of the nonlinear cost at every outer.
    """
    v_full = v_total + delta_v
    chi_full = chi_total + delta_chi
    J_b = 0.5 * jnp.sum(v_full * v_full)
    J_q = 0.5 * jnp.sum(chi_full * chi_full)
    T = x_l_traj.shape[0] - 1
    dx0 = apply_B_half(delta_v)
    deta = expand_eta(jax.vmap(apply_Q_half)(delta_chi), T)
    dx_traj = window_tlm_rollout_forced(tlm_op, x_l_traj, dx0, deta)

    def obs_term(i: ArrayLike) -> jax.Array:
        j = obs_window_indices[i]
        resid = Hs[i] @ dx_traj[j] - innovations[i]
        return 0.5 * jnp.sum(R_inv_diag * resid * resid)

    per_i = jax.vmap(obs_term)(jnp.arange(obs_window_indices.shape[0]))
    J_o = jnp.sum(jnp.where(obs_time_mask, per_i, 0.0))
    return J_b + J_q + J_o
