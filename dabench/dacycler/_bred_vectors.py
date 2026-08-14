"""Bred-vector construction of a climatological background covariance.

Physics-agnostic breeding (Toth & Kalnay 1997) for sampling the
growing-error subspace of *any* forecast model, used to build a
flow-aware low-rank ``B`` for the matrix-free 4D-Var cyclers.

The breeding cycle, run independently for each ensemble member along a
control trajectory sampled at the breeding cadence:

  1. Seed a perturbation ``p`` and rescale it to ``init_amplitude``.
  2. Grow both legs ``growth_steps`` model steps: the perturbed leg
     ``forecast_fn(x_ctrl + p)`` and the control leg ``x_ctrl_next``.
  3. The bred vector is ``d = perturbed - control`` (the grown error).
  4. Rescale ``d`` back to ``init_amplitude`` and feed it to the next
     anchor.  The first ``n_spinup`` cycles are discarded so the
     retained vectors have aligned with the local growing modes.

Collected bred vectors are stacked column-wise and reduced to
:class:`BFactors` via a thin SVD, matching the ``B = U sigma^2 U^T +
sigma_bg^2 (I - U U^T)`` convention of
:mod:`dabench.dacycler._var4d_operator_utils`.
"""

from typing import Callable

import jax
import jax.numpy as jnp
import numpy as np

from ._var4d_operator_utils import BFactors, finalize_lowrank_factors

ArrayLike = np.ndarray | jax.Array
# forecast_fn(x0: (D,), n_steps: int) -> x_after: (D,)
ForecastFn = Callable[[ArrayLike, int], ArrayLike]


def _amplitude(d: ArrayLike, norm: str) -> jax.Array:
    """Scalar size of ``d`` under the chosen breeding norm."""
    if norm == "rms":
        return jnp.sqrt(jnp.mean(d * d))
    if norm == "l2":
        return jnp.sqrt(jnp.sum(d * d))
    raise ValueError(f"norm must be 'rms' or 'l2'; got {norm!r}.")


def _rescale(d: ArrayLike, target: float, norm: str) -> jax.Array:
    """Rescale ``d`` so its ``norm`` equals ``target`` (safe at d=0)."""
    a = _amplitude(d, norm)
    return d * (target / jnp.where(a > 0, a, jnp.ones_like(a)))


def breed_vectors(
        forecast_fn: ForecastFn,
        control_traj: ArrayLike,
        *,
        init_amplitude: float,
        growth_steps: int,
        n_spinup: int = 3,
        ensemble_size: int = 8,
        seed: int = 0,
        norm: str = "rms",
        forecast_control: bool = True,
        ) -> tuple[jax.Array, dict]:
    """Grow an ensemble of bred vectors along a control trajectory.

    Args:
        forecast_fn: Deterministic model rollout
            ``forecast_fn(x0, growth_steps) -> x_after`` mapping a
            ``(D,)`` state to its state after ``growth_steps`` steps.
            Must be jax-traceable (it is driven inside ``lax.scan`` /
            ``vmap``).
        control_traj: Control states ``(n_anchor, D)`` sampled at the
            breeding cadence (consecutive rows ``growth_steps`` model
            steps apart).  Anchors ``control_traj[:-1]`` are bred from.
        init_amplitude: Target perturbation size (in ``norm``) used for
            the seed and for every rescale.
        growth_steps: Model steps per breeding interval (caller converts
            a wall-clock breeding window to steps via ``delta_t``).
        n_spinup: Leading breeding cycles discarded per member.
        ensemble_size: Independent breeding chains (different seeds).
        seed: PRNG seed for the perturbation seeds.
        norm: Breeding norm, ``"rms"`` (per-coefficient) or ``"l2"``.
        forecast_control: If True, grow the control leg with
            ``forecast_fn`` too (honest when ``forecast_fn`` differs
            from the model that generated ``control_traj``).  If False,
            reuse ``control_traj[1:]`` as the control leg (exact and
            cheaper when the trajectory is self-consistent).

    Returns:
        ``(bred, info)`` where ``bred`` is ``(M, D)`` with
        ``M = ensemble_size * (n_anchor - 1 - n_spinup)`` collected
        bred vectors and ``info`` is a JSON-friendly provenance dict.
    """
    control_traj = jnp.asarray(control_traj)
    if control_traj.ndim != 2 or control_traj.shape[0] < 2:
        raise ValueError(
                "control_traj must be 2-D with >= 2 anchors; got shape "
                f"{tuple(control_traj.shape)}.")
    n_anchor, D = control_traj.shape
    n_used = n_anchor - 1
    if not 0 <= n_spinup < n_used:
        raise ValueError(
                f"n_spinup={n_spinup} must satisfy 0 <= n_spinup < "
                f"{n_used} (n_anchor - 1).")
    dtype = control_traj.dtype
    gs = int(growth_steps)
    amp = float(init_amplitude)

    anchors = control_traj[:-1]                              # (n_used, D)
    if forecast_control:
        control_next = jax.vmap(lambda x: forecast_fn(x, gs))(anchors)
    else:
        control_next = control_traj[1:]

    def member_chain(p0: jax.Array) -> jax.Array:
        def step(p: jax.Array, inp: tuple[jax.Array, jax.Array]):
            x_ctrl, x_cn = inp
            x_pn = forecast_fn(x_ctrl + p, gs)
            d = x_pn - x_cn
            return _rescale(d, amp, norm), d
        _, ds = jax.lax.scan(step, p0, (anchors, control_next))
        return ds                                            # (n_used, D)

    keys = jax.random.split(jax.random.PRNGKey(int(seed)), int(ensemble_size))
    seeds = jax.vmap(
            lambda k: _rescale(jax.random.normal(k, (D,), dtype=dtype),
                               amp, norm))(keys)
    bred = jax.vmap(member_chain)(seeds)                     # (E, n_used, D)
    bred = bred[:, int(n_spinup):, :].reshape(-1, D)         # (M, D)

    info = {
            "n_anchor": int(n_anchor), "state_dim": int(D),
            "growth_steps": gs, "n_spinup": int(n_spinup),
            "ensemble_size": int(ensemble_size), "seed": int(seed),
            "norm": str(norm), "forecast_control": bool(forecast_control),
            "init_amplitude": amp, "n_bred_vectors": int(bred.shape[0]),
            }
    return bred, info


def bred_vectors_to_B_factors(
        bred: ArrayLike,
        K: int = 50,
        sigma_bg: float = 0.0,
        center: bool = False,
        target_trace: float | None = None,
        ) -> tuple[BFactors, dict]:
    """Reduce a stack of bred vectors to rank-``K`` :class:`BFactors`.

    The sample covariance of the ``(M, D)`` bred matrix is
    ``C = D_mat^T D_mat / (M - 1)``; its leading eigenpairs (the left
    singular factors of ``D_mat^T``) give the flow-dependent subspace.
    ``BFactors.sigma`` stores the *standard deviations* ``sqrt(eig)`` so
    that ``B = U sigma^2 U^T + sigma_bg^2 (I - U U^T)``.

    Args:
        bred: Bred vectors of shape ``(M, D)`` (rows are samples).
        K: Retained rank; capped at ``min(D, rank(bred))``.
        sigma_bg: Isotropic floor stored on the returned BFactors.
        center: Subtract the sample mean before the SVD.  Bred vectors
            are already anomalies, so this defaults to False.
        target_trace: If given, scale the spectrum so the retained
            subspace carries ``target_trace * variance_retained`` total
            variance (operational trace matching; pass
            ``D * sigma_bg**2`` to match an identity-B budget).

    Returns:
        ``(BFactors, info)`` with ``U: (D, K)``, ``sigma: (K,)``.
    """
    D_mat = jnp.asarray(bred)
    if D_mat.ndim != 2 or D_mat.shape[0] < 2:
        raise ValueError(
                f"bred must be 2-D with >= 2 rows; got {tuple(D_mat.shape)}.")
    M, D = D_mat.shape
    if center:
        D_mat = D_mat - jnp.mean(D_mat, axis=0, keepdims=True)
    U_full, s_full, _ = jnp.linalg.svd(D_mat.T, full_matrices=False)
    eig = (s_full ** 2) / float(M - 1)                       # (r,)
    K_eff = int(min(int(K), U_full.shape[1]))
    U_K = U_full[:, :K_eff]
    eig_K = eig[:K_eff]

    total_var = float(jnp.sum(eig))
    # Delegate the trace-match to the shared finalize (raw frame).  Passing
    # ``total_var`` (the FULL spectrum) reproduces the historical alpha2 =
    # target_trace / sum(eig_full) exactly, even for K < full rank.
    bf, _fin = finalize_lowrank_factors(
            U_K, jnp.sqrt(jnp.maximum(eig_K, 0.0)), int(D),
            sigma_bg=None, scale=None, target_trace=target_trace,
            total_var=total_var)
    alpha2 = _fin["alpha2"]
    bf = BFactors(U=bf.U, sigma=bf.sigma, sigma_bg=float(sigma_bg))
    info = {
            "n_bred_vectors": int(M), "state_dim": int(D),
            "K_requested": int(K), "K_effective": K_eff,
            "eigvals_top8": [float(v) for v in np.asarray(eig[:8])],
            "total_variance": total_var,
            "variance_retained": float(jnp.sum(eig_K)) / max(total_var, 1e-30),
            "alpha2": float(alpha2), "sigma_bg": float(sigma_bg),
            "target_trace": (None if target_trace is None
                             else float(target_trace)),
            "center": bool(center),
            }
    return bf, info


def build_bred_clim_B(
        forecast_fn: ForecastFn,
        control_traj: ArrayLike,
        *,
        K: int = 50,
        init_amplitude: float,
        growth_steps: int,
        n_spinup: int = 3,
        ensemble_size: int = 8,
        seed: int = 0,
        norm: str = "rms",
        forecast_control: bool = True,
        sigma_bg: float = 0.0,
        center: bool = False,
        target_trace: float | None = None,
        ) -> tuple[BFactors, dict]:
    """Breed vectors and reduce them to a climatological :class:`BFactors`.

    Convenience wrapper chaining :func:`breed_vectors` and
    :func:`bred_vectors_to_B_factors`.  See those functions for the
    full argument semantics.

    Returns:
        ``(BFactors, info)`` where ``info`` merges the breeding and
        factorisation provenance under ``"breeding"`` / ``"factors"``.
    """
    bred, breed_info = breed_vectors(
            forecast_fn, control_traj, init_amplitude=init_amplitude,
            growth_steps=growth_steps, n_spinup=n_spinup,
            ensemble_size=ensemble_size, seed=seed, norm=norm,
            forecast_control=forecast_control)
    bf, fac_info = bred_vectors_to_B_factors(
            bred, K=K, sigma_bg=sigma_bg, center=center,
            target_trace=target_trace)
    return bf, {"breeding": breed_info, "factors": fac_info}
