"""Utils for matrix-free (operator-based) 4D-Var cyclers.

These primitives are physics-agnostic: they accept a generic linear
operator callable ``apply_M(dx) -> dy`` (typically the tangent linear
model of some forecast at a fixed linearisation point) rather than
materialising the Jacobian.  They power
:class:`dabench.dacycler.Var4DOperator`, but can be composed
independently by users who supply their own outer loop.

The control-variable transform (CVT) preconditioning convention used
throughout: the background-error covariance is modelled as
``B = U sigma^2 U^T + sigma_bg^2 (I - U U^T)`` so its symmetric square
root is ``B^(1/2) = U sigma U^T + sigma_bg (I - U U^T)``, and the
inner-loop optimisation is performed in whitened control variables
``v`` related to the physical increment ``dx`` by ``dx = B^(1/2) v``.
"""

from typing import Any, Callable, NamedTuple

import jax
import jax.numpy as jnp
import numpy as np


# For typing
ArrayLike = np.ndarray | jax.Array
TLMOp = Callable[[ArrayLike], ArrayLike]


class BFactors(NamedTuple):
    """Rank-K factorisation of a background-error covariance.

    Represents ``B = U sigma^2 U^T + sigma_bg^2 (I - U U^T)``, whose
    matrix-free square root is recovered by :func:`build_B_half`.

    Attributes:
        U: Orthonormal columns of shape ``(system_dim, K)`` spanning
            the retained subspace.
        sigma: Singular values of shape ``(K,)``.
        sigma_bg: Isotropic floor applied to the ``U``-orthogonal
            complement (``>= 0``).
    """
    U: ArrayLike
    sigma: ArrayLike
    sigma_bg: float


def extract_B_factors(
        apply_M: TLMOp,
        system_dim: int,
        K: int = 50,
        E: int = 64,
        seed: int = 0,
        sigma_bg: float = 0.0,
        dtype: Any = jnp.float64,
        ) -> BFactors:
    """Randomised SVD of a matrix-free linear operator.

    Pushes ``E`` Gaussian probe directions through ``apply_M`` and
    takes the top-``K`` left singular factors of the resulting
    ``(system_dim, E)`` output stack.  By construction the sample
    covariance of the output equals ``M M^T`` in expectation, so its
    leading left singular vectors give an unstable-manifold basis at
    the operator's linearisation point — exactly what a rank-K B
    approximation needs.

    Args:
        apply_M: Forward linear operator. Must be strictly linear in
            its single argument.
        system_dim: Dimension ``D`` of the operator's input/output.
        K: Rank of the retained subspace. Must satisfy
            ``K <= min(system_dim, E)``.
        E: Number of Gaussian probes. ``E >= K`` is required;
            ``E >= 1.5 * K`` is recommended for spectral stability.
        seed: PRNG seed for the probes.
        sigma_bg: Isotropic floor stored on the returned BFactors.
        dtype: Working precision. ``jnp.float64`` is recommended.

    Returns:
        :class:`BFactors` with shapes ``U: (system_dim, K)`` and
        ``sigma: (K,)``.
    """
    if K > min(system_dim, E):
        raise ValueError(
                f"K={K} exceeds min(system_dim={system_dim}, E={E}).")
    rng = jax.random.PRNGKey(int(seed))
    Z = jax.random.normal(rng, (E, system_dim), dtype=dtype)
    Y = jax.vmap(apply_M)(Z)                          # (E, D)
    U_full, s_full, _ = jnp.linalg.svd(Y.T, full_matrices=False)
    return BFactors(U=U_full[:, :K], sigma=s_full[:K],
                    sigma_bg=float(sigma_bg))


def build_B_half(bf: BFactors) -> Callable[[ArrayLike], ArrayLike]:
    """Close BFactors into a matrix-free ``B^(1/2)`` operator.

    The returned closure computes
    ``B^(1/2) v = U sigma (U^T v) + sigma_bg (v - U U^T v)``,
    which is both the forward and adjoint action of the symmetric
    square root.  Cost per application is ``O(system_dim * K)``.

    Args:
        bf: Background-covariance factors from
            :func:`extract_B_factors`.

    Returns:
        Closure ``apply_B_half(v)`` mapping ``(system_dim,)`` to
        ``(system_dim,)``.
    """
    U, sigma, sigma_bg = bf.U, bf.sigma, float(bf.sigma_bg)

    def apply_B_half(v: ArrayLike) -> ArrayLike:
        UTv = U.T @ v                                       # (K,)
        return U @ ((sigma - sigma_bg) * UTv) + sigma_bg * v

    return apply_B_half


def finalize_lowrank_factors(
        U_raw: ArrayLike,
        sigma_raw: ArrayLike,
        state_dim: int,
        *,
        sigma_bg: float | None,
        scale: ArrayLike | None = None,
        target_trace: float | str | None = "auto",
        total_var: float | None = None,
        ) -> tuple[BFactors, dict]:
    """Trace-match (and, for ``scale``, congruence-map) raw low-rank factors.

    Shared finalize for any persisted raw square-root factor pair
    ``(U_raw, sigma_raw)`` (bred vectors, climatological ensemble anomalies,
    ...): the covariance is kept in SQUARE-ROOT form throughout and the
    control-variable-transform (CVT) congruence is applied to the anomaly
    factor, never by materialising a dense ``B``.

    Args:
        U_raw: Orthonormal columns ``(state_dim, K)`` of the raw factor.
        sigma_raw: Raw ``B^(1/2)`` singular values ``(K,)`` so that
            ``B_raw = U_raw diag(sigma_raw^2) U_raw^T``.
        state_dim: Physical dimension ``D``.
        sigma_bg: Isotropic background std used for trace matching.  The
            retained subspace is scaled so ``tr(B) = D * sigma_bg^2``.
        scale: If ``None`` the trace-match is performed in the raw frame in
            place.  Otherwise the congruence transform ``B_n = D^{-1} B_raw
            D^{-1}`` with ``D = diag(scale)`` is applied first (re-SVD of the
            scaled factor ``L = (U_raw / scale) * sigma_raw``).  Must have
            shape ``(state_dim,)``.
        target_trace: ``"auto"`` resolves to ``state_dim * sigma_bg^2``
            (``None`` when ``sigma_bg is None``); pass an explicit float/None
            to override.
        total_var: Optional precomputed total variance to trace-match
            against (raw frame only).  When given, ``alpha2 =
            target_trace / total_var`` rather than ``sum(sigma_raw^2)`` --
            lets callers trace-match a truncated subspace against the FULL
            spectrum (used by the bred path to stay byte-identical).

    Returns:
        ``(bf, info)`` where ``bf`` is a :class:`BFactors` with
        ``sigma_bg == 0`` and ``info`` carries ``alpha2``, ``eigvals_top8``,
        ``frame`` (``"raw"`` / ``"normalised"``), ``K_effective`` and the
        resolved ``target_trace``.
    """
    if target_trace == "auto":
        tt = (None if sigma_bg is None
              else float(state_dim) * float(sigma_bg) ** 2)
    else:
        tt = target_trace
    U_raw = jnp.asarray(U_raw)
    sigma_raw = jnp.asarray(sigma_raw)
    if scale is None:
        eig = sigma_raw ** 2
        tv = float(total_var) if total_var is not None else float(jnp.sum(eig))
        alpha2 = (1.0 if tt is None else float(tt) / max(tv, 1e-30))
        sigma = jnp.sqrt(jnp.maximum(eig, 0.0) * alpha2)
        bf = BFactors(U=U_raw, sigma=sigma, sigma_bg=0.0)
        eigvals_top8 = [float(v) for v in np.asarray(eig[:8])]
        frame = "raw"
    else:
        s = jnp.asarray(scale, dtype=U_raw.dtype)
        if s.shape != (state_dim,):
            raise ValueError(
                f"scale must have shape ({state_dim},); got {tuple(s.shape)}.")
        L = (U_raw / s[:, None]) * sigma_raw[None, :]           # (D, K)
        U_n, sv_n, _ = jnp.linalg.svd(L, full_matrices=False)
        eig_n = sv_n ** 2
        tv = float(jnp.sum(eig_n))
        alpha2 = (1.0 if tt is None else float(tt) / max(tv, 1e-30))
        sigma_n = jnp.sqrt(jnp.maximum(eig_n, 0.0) * alpha2)
        bf = BFactors(U=U_n, sigma=sigma_n, sigma_bg=0.0)
        eigvals_top8 = [float(v) for v in np.asarray(eig_n[:8])]
        frame = "normalised"
    info = {
            "alpha2": float(alpha2),
            "eigvals_top8": eigvals_top8,
            "frame": frame,
            "K_effective": int(bf.U.shape[1]),
            "target_trace": (None if tt is None else float(tt)),
            }
    return bf, info


def window_tlm_rollout(
        tlm_op: Callable[[ArrayLike, ArrayLike], ArrayLike],
        x_traj: ArrayLike,
        dx0: ArrayLike,
        ) -> jax.Array:
    """Propagate an increment along a fixed background trajectory.

    Args:
        tlm_op: Matrix-free TLM callable ``tlm_op(x_t, dx_t) -> dx_{t+1}``
            that is linear in its second argument.
        x_traj: Background trajectory of shape ``(T + 1, system_dim)``.
        dx0: Initial increment of shape ``(system_dim,)``.

    Returns:
        Increment trajectory of shape ``(T + 1, system_dim)`` with
        ``dx_traj[0] = dx0`` and
        ``dx_traj[t + 1] = tlm_op(x_traj[t], dx_traj[t])``.
    """
    T = x_traj.shape[0] - 1

    def step(dx_t: ArrayLike, x_t: ArrayLike
             ) -> tuple[jax.Array, jax.Array]:
        dx_next = tlm_op(x_t, dx_t)
        return dx_next, dx_next

    _, dx_tail = jax.lax.scan(step, dx0, x_traj[:T])
    return jnp.concatenate([dx0[None, :], dx_tail], axis=0)


def quadratic_cost(
        delta_v: ArrayLike,
        v_total: ArrayLike,
        tlm_op: Callable[[ArrayLike, ArrayLike], ArrayLike],
        x_b_traj: ArrayLike,
        Hs: ArrayLike,
        innovations: ArrayLike,
        obs_window_indices: ArrayLike,
        obs_time_mask: ArrayLike,
        R_inv_diag: ArrayLike,
        apply_B_half: Callable[[ArrayLike], ArrayLike],
        ) -> jax.Array:
    """Incremental 4D-Var cost in CVT v-space.

    Computes
    ``J(delta_v) = 0.5 |v_total + delta_v|^2
        + 0.5 sum_i (H_i dx_{j(i)} - d_i)^T R^{-1} (H_i dx_{j(i)} - d_i)``
    where ``j(i) = obs_window_indices[i]`` maps each obs to its
    trajectory step, ``dx_{j(i)}`` is the TLM-propagated increment at
    that step starting from ``dx_0 = B^(1/2) (v_total + delta_v)``,
    and ``d_i = y_obs_i - H_i x_b_traj_{j(i)}`` are precomputed
    innovations against the current outer's nonlinear trajectory.

    The obs-indexed convention matches :class:`Var4D._innerloop_4d`:
    each obs row in ``Hs`` / ``innovations`` is associated with a
    trajectory step via ``obs_window_indices``, not with a timestep
    directly.

    Args:
        delta_v: Inner-loop increment in control variable space,
            shape ``(system_dim,)``. Differentiable.
        v_total: Accumulated control variable from previous outer
            iterations.
        tlm_op: Matrix-free TLM ``tlm_op(x_t, dx_t) -> dx_{t+1}``.
        x_b_traj: Background trajectory of shape
            ``(window_steps + 1, system_dim)``.
        Hs: Per-obs linear observation operators of shape
            ``(n_obs, obs_dim, system_dim)``.
        innovations: Precomputed innovations of shape
            ``(n_obs, obs_dim)``.
        obs_window_indices: Trajectory step for each obs, shape
            ``(n_obs,)``, integer.
        obs_time_mask: Boolean mask of shape ``(n_obs,)``; obs at
            masked-out indices contribute zero.
        R_inv_diag: Diagonal of ``R^{-1}``, broadcast against an
            observation vector.
        apply_B_half: ``B^(1/2)`` preconditioner from
            :func:`build_B_half`.

    Returns:
        Scalar cost ``J(delta_v)``. Gradient w.r.t. ``delta_v`` is
        the negative right-hand side of the inner-loop normal
        equation; the HVP recovers the Gauss-Newton Hessian.
    """
    v_full = v_total + delta_v
    J_b = 0.5 * jnp.sum(v_full * v_full)
    dx0 = apply_B_half(v_full)
    dx_traj = window_tlm_rollout(tlm_op, x_b_traj, dx0)

    def obs_term(i: ArrayLike) -> jax.Array:
        j = obs_window_indices[i]
        H_dx = Hs[i] @ dx_traj[j]
        resid = H_dx - innovations[i]
        return 0.5 * jnp.sum(R_inv_diag * resid * resid)

    n_obs = obs_window_indices.shape[0]
    per_i = jax.vmap(obs_term)(jnp.arange(n_obs))
    J_o = jnp.sum(jnp.where(obs_time_mask, per_i, 0.0))
    return J_b + J_o


def pcg_lanczos_solve(
        hvp_fn: Callable[[ArrayLike], ArrayLike],
        b: ArrayLike,
        max_iter: int = 30,
        tol: float = 1e-6,
        verbose: bool = False,
        verbose_tag: str = "pcg",
        ) -> tuple[jax.Array, dict]:
    """Preconditioned CG with Lanczos diagnostics, fully jit-traceable.

    Solves ``H x = b`` for a s.p.d. operator ``H`` supplied as a
    matrix-free Hessian-vector product callable.  Preconditioning is
    implicit in the CVT: searching in whitened ``v`` coordinates makes
    the background-term Hessian the identity, so plain CG already
    behaves like preconditioned CG in the physical ``x``-space.

    Implemented as a ``jax.lax.while_loop`` so the iteration count can
    depend on the traced residual without breaking ``jit`` / ``scan``.
    Diagnostics are returned as fixed-length jax arrays of length
    ``max_iter`` (unfilled entries are zero); the ``alphas`` / ``betas``
    arrays together with the Lanczos identity yield the inner-loop
    tridiagonal for post-hoc Ritz / condition-number analysis.

    Args:
        hvp_fn: Hessian-vector product callable ``H p -> H @ p``.
        b: Right-hand side, shape ``(system_dim,)``.
        max_iter: Maximum CG iterations (Python int; fixes loop budget).
        tol: Relative residual tolerance for early stopping.

    Returns:
        Tuple ``(x, info)``.  ``info`` keys: ``alphas``, ``betas``,
        ``residual_norms``, ``n_iter``, ``converged``, ``breakdown``.
    """
    dtype = jnp.asarray(b).dtype
    M = int(max_iter)
    tol2 = jnp.asarray(float(tol) ** 2, dtype=dtype)
    rho0 = jnp.vdot(b, b)
    init = dict(
            x=jnp.zeros_like(b),
            r=b,
            p=b,
            rho_old=rho0,
            alphas=jnp.zeros(M, dtype=dtype),
            betas=jnp.zeros(M, dtype=dtype),
            res_norms=jnp.zeros(M + 1, dtype=dtype).at[0].set(
                    jnp.sqrt(rho0)),
            n_iter=jnp.int32(0),
            converged=jnp.bool_(False),
            breakdown=jnp.bool_(False),
            )
    target_sq = tol2 * rho0
    max_iter_arr = jnp.int32(M)

    def cond_fn(s):
        return ((~s['converged'])
                & (~s['breakdown'])
                & (s['n_iter'] < max_iter_arr))

    def body_fn(s):
        k = s['n_iter']
        Hp = hvp_fn(s['p'])
        pHp = jnp.vdot(s['p'], Hp)
        breakdown = pHp <= 0.0
        pHp_safe = jnp.where(breakdown, jnp.ones_like(pHp), pHp)
        alpha = s['rho_old'] / pHp_safe
        x_new = s['x'] + alpha * s['p']
        r_new = s['r'] - alpha * Hp
        rho_new = jnp.vdot(r_new, r_new)
        rho_old_safe = jnp.where(
                s['rho_old'] > 0,
                s['rho_old'],
                jnp.ones_like(s['rho_old']))
        beta = rho_new / rho_old_safe
        p_new = r_new + beta * s['p']
        res_norm = jnp.sqrt(jnp.maximum(rho_new, 0.0))
        converged = rho_new <= target_sq
        return dict(
                x=jnp.where(breakdown, s['x'], x_new),
                r=jnp.where(breakdown, s['r'], r_new),
                p=jnp.where(breakdown, s['p'], p_new),
                rho_old=jnp.where(breakdown, s['rho_old'], rho_new),
                alphas=s['alphas'].at[k].set(alpha),
                betas=s['betas'].at[k].set(beta),
                res_norms=s['res_norms'].at[k + 1].set(res_norm),
                n_iter=k + 1,
                converged=converged,
                breakdown=breakdown,
                )

    final = jax.lax.while_loop(cond_fn, body_fn, init)
    if verbose:
        last_idx = jnp.maximum(final['n_iter'] - 1, 0)
        jax.debug.print(
                "  [{tag}] n_iter={n}  conv={c}  brk={b}  "
                "||r0||={r0:.3e}  ||r_final||={rf:.3e}  "
                "rel={rel:.3e}",
                tag=verbose_tag,
                n=final['n_iter'],
                c=final['converged'],
                b=final['breakdown'],
                r0=final['res_norms'][0],
                rf=final['res_norms'][last_idx + 1],
                rel=final['res_norms'][last_idx + 1]
                    / jnp.where(final['res_norms'][0] > 0,
                                final['res_norms'][0],
                                jnp.ones_like(final['res_norms'][0])))
    info = {
            "alphas": final['alphas'],
            "betas": final['betas'],
            "residual_norms": final['res_norms'],
            "n_iter": final['n_iter'],
            "converged": final['converged'],
            "breakdown": final['breakdown'],
            }
    return final['x'], info
