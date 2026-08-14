"""Climatological ensemble-covariance background-error B in square-root form.

Builds a TRUE low-rank climatological background-error covariance ``B`` from
the mean-removed forecast-ensemble anomalies pooled over every
``(cycle, cycle_timestep)`` row of an ensemble DA run, and keeps it in
SQUARE-ROOT form throughout -- a running factor ``L`` such that ``B_bar approx
L L^T``.  The ``state_dim x state_dim`` covariance is NEVER materialised (that
would square the condition number and blow up memory); instead the anomaly
stack is reduced by a streaming rank-capped thin-SVD, exactly mirroring the
bred path (:mod:`dabench.dacycler._bred_vectors`), which SVDs the anomaly
stack ``D_mat.T`` directly.

Unlike a per-coefficient diagonal climatology, this retains the full
off-diagonal covariance structure as a low-rank ``B``.  The run-dependent
trace-match and (for a normalised-frame model) the control-variable-transform
(CVT) congruence ``B_n = D^{-1} B_raw D^{-1}`` are applied at LOAD time by the
shared :func:`dabench.dacycler.finalize_lowrank_factors` -- the SAME finalize
the bred path uses -- so ``B`` is formed only implicitly inside 4D-Var.

Public API
----------
EnsCovAccumulator(state_dim, max_rank)
    Streaming rank-capped thin-SVD of the forecast-ensemble anomaly stack.
save_ens_cov_factors(path, U, sigma, n_rows, n_ens, state_dim, info)
    Persist the raw square-root factors (schema ``"ens-cov"``).
load_ens_cov_b_half(path, *, sigma_bg, scale=None, K=None, ...)
    Reload raw factors -> trace-matched (CVT-aware) low-rank ``B^(1/2)`` op.
"""

from typing import Callable, Tuple

import numpy as np
import jax.numpy as jnp

from ._var4d_operator_utils import (
    BFactors, build_B_half, finalize_lowrank_factors)


class EnsCovAccumulator:
    """Streaming rank-capped thin-SVD of the forecast-ensemble anomaly stack.

    Keeps the climatological ensemble ``B`` in SQUARE-ROOT form -- a running
    factor ``L`` such that ``B_bar approx L L^T`` -- and NEVER forms the
    ``state_dim x state_dim`` Gram matrix ``C = sum Xc^T Xc`` (that would
    square the condition number and materialise a B-sized object).  This
    mirrors the bred path, which SVDs the anomaly stack ``D_mat.T`` directly.

    Each ``update(Xc)`` ingests one ``(cycle, cycle_timestep)`` row's FORECAST
    (background) ensemble ``Xc`` of shape ``(n_ens, state_dim)``, removes its
    ensemble mean, and scales the anomalies by ``1/sqrt(n_ens - 1)`` so that
    the row's contribution ``a^T a`` is exactly its unbiased sample covariance.
    The pooled climatological covariance over ``R`` rows is ``B_bar =
    (1/R) sum_rows cov_row = A^T A`` where ``A`` is the vertical stack of the
    per-row anomaly blocks each additionally scaled by ``1/sqrt(R)``.  Since
    ``R`` is only known at the end, rows are accumulated UNSCALED-by-R via
    incremental thin-SVD and the ``1/sqrt(R)`` pooling is folded into
    :meth:`factors` once ``R`` is final.

    Incremental step (rank-capped, exact until truncation bites): the running
    factor ``F = diag(s) U^T`` of shape ``(k, D)`` satisfies ``A^T A = F^T F``
    for the anomalies ``A`` seen so far.  Adding a new block ``a (m, D)`` gives
    ``A_new^T A_new = F^T F + a^T a = [F; a]^T [F; a]``, so the new factor is
    obtained by re-SVDing the small stack ``G = [F; a]`` of shape
    ``(k + m, D)`` and keeping its top-``max_rank`` right singular
    vectors/values.  Memory is ``O((max_rank + n_ens) * D)``; the full
    ``(R * n_ens, D)`` anomaly history is never held at once.
    """

    def __init__(self, state_dim: int, max_rank: int):
        self.state_dim = int(state_dim)
        self.max_rank = int(max_rank)
        self.n_rows = 0
        self.n_ens = 0
        self._U = np.zeros((self.state_dim, 0), dtype=np.float64)   # (D, k)
        self._s = np.zeros((0,), dtype=np.float64)                  # (k,)

    def update(self, Xc: "np.ndarray") -> None:
        """Ingest one row's forecast ensemble ``Xc`` of shape ``(n_ens, D)``."""
        X = np.asarray(Xc, dtype=np.float64)
        if X.ndim != 2 or X.shape[1] != self.state_dim:
            raise ValueError(
                f"row must be (n_ens, {self.state_dim}); got {tuple(X.shape)}.")
        m = X.shape[0]
        if m < 2:
            return                                       # no anomaly info
        self.n_ens = m
        self.n_rows += 1
        a = (X - X.mean(axis=0, keepdims=True)) / np.sqrt(m - 1.0)  # (m, D)

        # Running factor F = diag(s) U^T (k, D); A^T A = F^T F.  New anomalies
        # a extend the stack: A_new^T A_new = [F; a]^T [F; a].  Re-SVD the
        # small (k + m, D) stack and keep the top-max_rank right factors.
        F = (self._s[:, None] * self._U.T) if self._s.size else \
            np.zeros((0, self.state_dim), dtype=np.float64)
        G = np.concatenate([F, a], axis=0)               # (k + m, D)
        _, sc, Vt = np.linalg.svd(G, full_matrices=False)
        r = min(self.max_rank, sc.shape[0])
        self._U = np.ascontiguousarray(Vt[:r].T)         # (D, r)
        self._s = sc[:r]

    def factors(self) -> "Tuple[np.ndarray, np.ndarray]":
        """Return raw ``(U, sigma)`` of ``B_bar approx U diag(sigma^2) U^T``.

        ``sigma`` are the square-root singular values pooled over all rows
        (the ``1/sqrt(n_rows)`` folded in here); ``U`` is orthonormal
        ``(state_dim, K_eff)``.  These are the UN-rescaled raw factors
        consumed by :func:`save_ens_cov_factors` (no trace-match / CVT).
        """
        if self.n_rows < 1:
            raise ValueError("no rows accumulated; nothing to factor.")
        sigma = self._s / np.sqrt(float(self.n_rows))
        return np.asarray(self._U, dtype=np.float64), np.asarray(sigma)


def save_ens_cov_factors(
    path: str, U_raw: "np.ndarray", sigma_raw: "np.ndarray",
    n_rows: int, n_ens: int, state_dim: int, info: dict,
) -> None:
    """Persist raw square-root factors of the climatological ensemble ``B``.

    ``(U_raw, sigma_raw)`` are the un-rescaled factors from
    :meth:`EnsCovAccumulator.factors` -- ``U_raw`` orthonormal
    ``(state_dim, K)`` and ``sigma_raw`` the ``B^(1/2)`` singular values, so
    ``B_bar approx U_raw diag(sigma_raw^2) U_raw^T``.  They come from a
    streaming thin-SVD of the mean-removed FORECAST (background) ensemble
    anomalies pooled over every ``(cycle, cycle_timestep)`` row -- the SQUARE
    ROOT is kept throughout; the ``state_dim x state_dim`` covariance is NEVER
    formed.  Stored UN-rescaled -- NO trace-match, NO CVT congruence baked in
    -- so the run-dependent ``sigma_bg`` / frame ``scale`` are applied later by
    :func:`load_ens_cov_b_half` via the SAME
    :func:`dabench.dacycler.finalize_lowrank_factors` the bred path uses
    (which re-SVDs the CVT-scaled anomaly factor, forming ``B`` only
    implicitly).  A ``schema="ens-cov"`` tag disambiguates the npz.
    """
    import json
    import os

    if int(n_ens) < 2 or int(n_rows) < 1:
        raise ValueError(
            f"ens-cov needs n_rows>=1 and n_ens>=2; got n_rows={n_rows}, "
            f"n_ens={n_ens}.")
    U = np.asarray(U_raw, dtype=np.float64)
    sigma = np.asarray(sigma_raw, dtype=np.float64)

    out_dir = os.path.dirname(os.path.abspath(path))
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    np.savez(
        path,
        U=U,
        sigma=sigma,
        state_dim=np.int64(state_dim),
        n_rows=np.int64(n_rows),
        n_ens=np.int64(n_ens),
        schema=np.str_("ens-cov"),
        info_json=json.dumps(info),
    )


def _hybridize_ens_cov_factors(bf, sigma_bg: float, alpha: float, beta: float):
    """Additive low-rank hybrid ``B = alpha·sigma_bg^2 I + beta·B_ens``.

    ``bf`` is a trace-matched :class:`BFactors` (from
    :func:`dabench.dacycler.finalize_lowrank_factors`, so ``sigma_bg == 0`` and
    ``sum(sigma^2) = state_dim·sigma_bg^2``).  Inside ``U`` the eigenvalue
    becomes ``alpha·sigma_bg^2 + beta·sigma_ens^2``; on the ``U``-complement
    the isotropic floor is ``alpha·sigma_bg^2`` (so ``sigma_bg_h = sqrt(alpha)·
    sigma_bg``).  ``(alpha=0, beta=1)`` ⇒ pure trace-matched ens-cov B;
    ``(alpha=1, beta=0)`` ⇒ identity-B at ``sigma_bg`` (``sigma_h == sigma_bg``
    and floor ``sigma_bg`` ⇒ ``build_B_half`` collapses to ``v → sigma_bg·v``).
    """
    a = float(alpha)
    b = float(beta)
    sb2 = float(sigma_bg) ** 2
    sigma_h = jnp.sqrt(jnp.maximum(a * sb2 + b * bf.sigma ** 2, 0.0))
    return BFactors(U=bf.U, sigma=sigma_h,
                    sigma_bg=float(np.sqrt(max(a, 0.0))) * float(sigma_bg))


def load_ens_cov_b_half(
    path: str,
    *,
    sigma_bg: float,
    scale: "np.ndarray | None" = None,
    K: "int | None" = None,
    hybrid_alpha: float = 0.0,
    hybrid_beta: float = 1.0,
) -> Tuple[Callable, dict]:
    """Reload raw ens-cov factors → TRUE low-rank ``B^(1/2)`` (trace-matched).

    Loads the raw ``(U, sigma)`` written by :func:`save_ens_cov_factors`,
    optionally truncates to the leading ``K`` eigenmodes, then applies the
    run-dependent trace-match (``tr(B) = state_dim · sigma_bg^2``) -- and, for
    a normalised-frame model (``scale != None``), the CVT congruence transform
    -- via :func:`dabench.dacycler.finalize_lowrank_factors` (the SAME finalize
    the bred path uses).  Finally the additive low-rank hybrid ``B =
    alpha·sigma_bg^2 I + beta·B_ens`` is formed by
    :func:`_hybridize_ens_cov_factors`, giving total ``tr(B) =
    (alpha+beta)·state_dim·sigma_bg^2``.  Returns ``(B_half_op, info)``.
    """
    import json

    d = np.load(path, allow_pickle=False)
    schema = str(d["schema"]) if "schema" in d.files else "missing"
    if schema != "ens-cov":
        raise ValueError(
            f"{path} is not an ens-cov factor npz (schema={schema!r}); "
            f"use the matching loader.")
    U_raw = np.asarray(d["U"])
    sigma_raw = np.asarray(d["sigma"])
    state_dim = int(d["state_dim"])
    build_info = json.loads(str(d["info_json"]))

    K_req = None if K is None else int(K)
    if K_req is not None and K_req > 0:
        U_raw = U_raw[:, :K_req]
        sigma_raw = sigma_raw[:K_req]

    bf, _fin = finalize_lowrank_factors(
        U_raw, sigma_raw, state_dim, sigma_bg=sigma_bg, scale=scale)
    alpha2 = _fin["alpha2"]
    eigvals_top8 = _fin["eigvals_top8"]
    frame = _fin["frame"]
    bf_hyb = _hybridize_ens_cov_factors(
        bf, float(sigma_bg), float(hybrid_alpha), float(hybrid_beta))
    B_half_op = build_B_half(bf_hyb)

    K_eff = int(bf_hyb.U.shape[1])
    floor2 = float(bf_hyb.sigma_bg) ** 2
    trace_B = float(np.sum(np.asarray(bf_hyb.sigma) ** 2)
                    + max(state_dim - K_eff, 0) * floor2)
    info = {
        "method": "ens-cov-lowrank",
        "frame": frame,
        "source_npz": str(path),
        "state_dim": state_dim,
        "K_effective": K_eff,
        "K_requested": (0 if K_req is None else K_req),
        "eigvals_top8": eigvals_top8,
        "alpha2": float(alpha2),
        "hybrid_alpha": float(hybrid_alpha),
        "hybrid_beta": float(hybrid_beta),
        "sigma_bg_target": float(sigma_bg),
        "trace_B": trace_B,
        "n_rows_accum": build_info.get("n_rows_accum"),
        "ensemble_dim": build_info.get("ensemble_dim"),
        "src_method": build_info.get("src_method",
                                     build_info.get("method")),
        "model_label": build_info.get("model_label"),
    }
    return B_half_op, info
