"""Tests for the climatological ensemble-covariance B construction.

Exercises :mod:`dabench.dacycler._ensemble_cov`:

  * :class:`EnsCovAccumulator` — the streaming rank-capped thin-SVD keeps
    the SQUARE ROOT of the pooled forecast-ensemble covariance exactly
    (full-rank stream == dense pooled B_bar) and recovers the leading
    eigenpairs under a rank cap.
  * :func:`save_ens_cov_factors` / :func:`load_ens_cov_b_half` — raw-frame
    and normalised-frame (CVT) trace-match, and the additive low-rank hybrid
    identity/pure-ens limits, verified through :func:`build_B_half`.
"""

import jax

jax.config.update("jax_enable_x64", True)

import numpy as np  # noqa: E402

from dabench.dacycler import (  # noqa: E402
        EnsCovAccumulator,
        save_ens_cov_factors,
        load_ens_cov_b_half,
        localize_lowrank_factors,
        )


def _pooled_B(rows):
    """Dense pooled unbiased sample covariance over a list of (n_ens, D) rows."""
    covs = []
    for X in rows:
        a = X - X.mean(axis=0, keepdims=True)
        covs.append(a.T @ a / (X.shape[0] - 1.0))
    return np.mean(covs, axis=0)


def _make_rows(D=8, n_ens=6, R=5, seed=0):
    rng = np.random.default_rng(seed)
    return [rng.normal(size=(n_ens, D)) for _ in range(R)]


def test_full_rank_stream_matches_dense_pooled():
    D, n_ens, R = 8, 6, 5
    rows = _make_rows(D, n_ens, R)
    acc = EnsCovAccumulator(D, max_rank=D)
    for X in rows:
        acc.update(X)
    U, sig = acc.factors()
    B_stream = U @ np.diag(sig ** 2) @ U.T
    B_bar = _pooled_B(rows)
    rel = np.linalg.norm(B_stream - B_bar) / np.linalg.norm(B_bar)
    assert rel < 1e-10, rel


def test_rank_cap_low_rank_approximation():
    # Per-step truncation of the streaming SVD does not reproduce the exact
    # top-k eigenpairs of the full pooled covariance (early rows are truncated
    # before later anomalies can interact with them), but it must (a) keep the
    # requested rank, (b) approximate the leading eigenvalue well, and (c) not
    # exceed the retained variance of the exact leading subspace.
    D, n_ens, R = 8, 6, 5
    rows = _make_rows(D, n_ens, R, seed=1)
    B_bar = _pooled_B(rows)
    w = np.sort(np.linalg.eigvalsh(B_bar))[::-1]

    acc = EnsCovAccumulator(D, max_rank=3)
    for X in rows:
        acc.update(X)
    U, sig = acc.factors()
    assert U.shape[1] == 3
    got = np.sort(sig ** 2)[::-1]
    # Leading eigenvalue captured to a few percent; columns orthonormal.
    assert abs(got[0] - w[0]) / w[0] < 0.1, (got[0], w[0])
    assert np.allclose(U.T @ U, np.eye(3), atol=1e-10)
    # Retained variance bounded by the exact top-3 (no energy invented).
    assert got.sum() <= w[:3].sum() + 1e-8, (got.sum(), w[:3].sum())


def _save_tmp(tmp_path, rows, D, n_ens, R):
    acc = EnsCovAccumulator(D, max_rank=D)
    for X in rows:
        acc.update(X)
    U, sig = acc.factors()
    npz = str(tmp_path / "ens_cov.npz")
    save_ens_cov_factors(npz, U, sig, R, n_ens, D,
                         {"method": "test", "n_rows_accum": R,
                          "ensemble_dim": n_ens})
    return npz


def _dense_B_from_op(op, D):
    return np.stack([np.asarray(op(np.asarray(op(np.eye(D)[i]))))
                     for i in range(D)], axis=1)


def test_roundtrip_raw_trace_match(tmp_path):
    D, n_ens, R = 8, 6, 5
    rows = _make_rows(D, n_ens, R, seed=2)
    npz = _save_tmp(tmp_path, rows, D, n_ens, R)
    op, info = load_ens_cov_b_half(npz, sigma_bg=0.1)
    assert info["frame"] == "raw"
    B = _dense_B_from_op(op, D)
    assert np.isclose(np.trace(B), D * 0.1 ** 2, atol=1e-9), np.trace(B)


def test_roundtrip_normalised_cvt(tmp_path):
    D, n_ens, R = 8, 6, 5
    rng = np.random.default_rng(3)
    rows = _make_rows(D, n_ens, R, seed=3)
    scale = 0.5 + rng.random(D)
    npz = _save_tmp(tmp_path, rows, D, n_ens, R)
    op, info = load_ens_cov_b_half(npz, sigma_bg=0.1, scale=scale)
    assert info["frame"] == "normalised"
    B = _dense_B_from_op(op, D)
    assert np.isclose(np.trace(B), D * 0.1 ** 2, atol=1e-9), np.trace(B)


def test_hybrid_identity_limit(tmp_path):
    D, n_ens, R = 8, 6, 5
    rows = _make_rows(D, n_ens, R, seed=4)
    npz = _save_tmp(tmp_path, rows, D, n_ens, R)
    op, _ = load_ens_cov_b_half(npz, sigma_bg=0.3,
                                hybrid_alpha=1.0, hybrid_beta=0.0)
    v = np.random.default_rng(9).normal(size=D)
    assert np.allclose(np.asarray(op(v)), 0.3 * v, atol=1e-10)


def test_hybrid_default_is_pure_ens(tmp_path):
    D, n_ens, R = 8, 6, 5
    rows = _make_rows(D, n_ens, R, seed=5)
    npz = _save_tmp(tmp_path, rows, D, n_ens, R)
    op_h, _ = load_ens_cov_b_half(npz, sigma_bg=0.1,
                                  hybrid_alpha=0.0, hybrid_beta=1.0)
    op_p, _ = load_ens_cov_b_half(npz, sigma_bg=0.1)
    v = np.random.default_rng(10).normal(size=D)
    assert np.allclose(np.asarray(op_h(v)), np.asarray(op_p(v)), atol=1e-12)


def test_localize_khatri_rao_matches_dense_schur():
    # (X X^T) o (G G^T) == X_loc X_loc^T exactly (machine precision).
    rng = np.random.default_rng(7)
    D, K, P = 20, 5, 4
    U, _ = np.linalg.qr(rng.normal(size=(D, K)))
    sig = np.abs(rng.normal(size=K)) + 0.1
    G = rng.normal(size=(D, P))
    X = U * sig[None, :]
    B_schur = (X @ X.T) * (G @ G.T)
    U_loc, sig_loc = localize_lowrank_factors(U, sig, G, K_cap=K * P)
    B_loc = (U_loc * sig_loc[None, :]) @ (U_loc * sig_loc[None, :]).T
    assert np.linalg.norm(B_loc - B_schur) / np.linalg.norm(B_schur) < 1e-12


def test_localize_preserves_trace_through_loader(tmp_path):
    # Localization changes structure, not total variance: the finalize trace-
    # match yields the SAME trace_B with or without localization.
    D, n_ens, R = 12, 8, 6
    rows = _make_rows(D, n_ens, R, seed=8)
    npz = _save_tmp(tmp_path, rows, D, n_ens, R)
    rng = np.random.default_rng(11)
    G = rng.normal(size=(D, 5))
    G = G / np.maximum(np.sqrt((G ** 2).sum(1, keepdims=True)), 1e-30)
    _, i0 = load_ens_cov_b_half(npz, sigma_bg=0.2,
                                hybrid_alpha=1.0, hybrid_beta=1.0)
    op1, i1 = load_ens_cov_b_half(npz, sigma_bg=0.2,
                                  hybrid_alpha=1.0, hybrid_beta=1.0,
                                  localize_G=G, localize_K=D)
    assert i0["localized"] is False and i1["localized"] is True
    assert i1["localize_P"] == 5
    assert np.isclose(i0["trace_B"], i1["trace_B"], rtol=1e-9)
    v = rng.normal(size=D)
    assert np.isfinite(np.asarray(op1(v))).all()
