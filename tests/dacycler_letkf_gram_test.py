"""LETKF dense-path Gram formulation (``local_gram``) vs legacy direct path.

The Gram path precomputes ``W = rinv_o vec(Y_o Y_o^T)`` and ``v = rinv_o d_o
Y_o`` once and forms each grid block's local moments by GEMM with the taper
rows, so no ``(chunk, K, n_obs)`` intermediate exists.  Must match "direct"
to round-off, incl. gradients, masks, multi-field, 4D stacking, ring fallback.
"""
import numpy as np
import pytest
import jax
import jax.numpy as jnp
from dabench.dacycler import LETKF

jax.config.update("jax_enable_x64", True)


def _geom(rng, G, n):
    ll = lambda m: np.stack([rng.uniform(-80, 80, m), rng.uniform(0, 360, m)], 1)
    return jnp.asarray(ll(G)), jnp.asarray(ll(n))


def _make(G, ens, chunk, mode, grid_ll=None, obs_ll=None, radius=None, **kw):
    if radius is None:
        radius = 3000.0 if grid_ll is not None else 3.0
    return LETKF(system_dim=G, delta_t=0.01, ensemble_dim=ens, model_obj=None,
                 grid_latlon=grid_ll, obs_latlon=obs_ll, localize_radius=radius,
                 grid_chunk=chunk, local_gram=mode, **kw)


def _rel(a, b):
    a, b = np.asarray(a), np.asarray(b)
    return np.max(np.abs(a - b)) / max(np.max(np.abs(b)), 1e-300)


def _inputs(rng, G, n_obs, ens, mask_frac=0.0):
    Xb = jnp.asarray(rng.standard_normal((G, ens)))
    Yb = jnp.asarray(rng.standard_normal((n_obs, ens)))
    Y = jnp.asarray(rng.standard_normal(n_obs))
    rinv = rng.uniform(0.5, 2.0, n_obs)
    if mask_frac:
        rinv[rng.uniform(size=n_obs) < mask_frac] = 0.0   # inactive obs
    return Xb, Yb, Y, jnp.asarray(rinv)


@pytest.mark.parametrize("chunk", [None, 1, 5, 8, 100])
@pytest.mark.parametrize("n_times,mask", [(1, 0.0), (3, 0.0), (3, 0.4)])
def test_gram_matches_direct_localized_analysis(chunk, n_times, mask):
    """Full analysis (incl. RTPS + multiplicative rho), stationary (n_times=1)
    and window-stacked/non-stationary masked networks, chunked/unchunked
    (dense chunked path streams taper rows)."""
    rng = np.random.default_rng(11)
    G, n_slice, ens = 13, 9, 4
    grid_ll, obs_ll = _geom(rng, G, n_slice)
    n_obs = n_slice * n_times
    Xb, Yb, Y, rinv = _inputs(rng, G, n_obs, ens, mask)
    idx = jnp.asarray(np.tile(np.arange(n_slice), n_times))
    out = {}
    for mode in ("direct", "gram"):
        c = _make(G, ens, chunk, mode, grid_ll, obs_ll,
                  rtps_relaxation=0.5, rtpp_relaxation=0.2)
        out[mode] = c._localized_analysis(Xb, Yb, Y, rinv, idx, rho=1.1)
    assert _rel(out["gram"], out["direct"]) <= 1e-10


def test_gram_matches_direct_multifield():
    """Multi-field state: several fields share one grid's per-point weights
    via to_grid/from_grid stacking (2 fields on the same grid points)."""
    rng = np.random.default_rng(5)
    G, n_obs, ens = 10, 12, 5
    grid_ll1, obs_ll = _geom(rng, G, n_obs)
    grid_ll = jnp.concatenate([grid_ll1, grid_ll1], 0)   # 2 fields
    Xb, Yb, Y, rinv = _inputs(rng, 2 * G, n_obs, ens, 0.3)
    idx = jnp.arange(n_obs)
    res = [_make(2 * G, ens, 6, m, grid_ll, obs_ll)._localized_analysis(
        Xb, Yb, Y, rinv, idx, rho=1.0) for m in ("direct", "gram")]
    assert _rel(res[1], res[0]) <= 1e-10


@pytest.mark.parametrize("chunk", [None, 3])
def test_gram_matches_direct_ring(chunk):
    """1-D index-ring fallback (no grid_latlon)."""
    rng = np.random.default_rng(2)
    G, n_obs, ens = 11, 7, 4
    Xb, Yb, Y, rinv = _inputs(rng, G, n_obs, ens, 0.2)
    idx = jnp.asarray(rng.choice(G, n_obs, replace=False))
    res = [_make(G, ens, chunk, m)._localized_analysis(
        Xb, Yb, Y, rinv, idx, rho=1.0) for m in ("direct", "gram")]
    assert _rel(res[1], res[0]) <= 1e-10


def test_gram_gradient_matches_direct():
    rng = np.random.default_rng(23)
    G, n_obs, ens = 13, 20, 4
    grid_ll, obs_ll = _geom(rng, G, n_obs)
    Xb, Yb, Y, rinv = _inputs(rng, G, n_obs, ens, 0.25)
    idx = jnp.arange(n_obs)

    def loss(mode, xb, yb):
        # Global radius: every point sees obs (an obs-free point has the
        # degenerate A=(K-1)I whose eigh VJP is NaN on BOTH paths).
        c = _make(G, ens, 5, mode, grid_ll, obs_ll, radius=12000.0)
        return jnp.sum(c._localized_analysis(xb, yb, Y, rinv, idx, rho=1.0) ** 2)

    for argn in (1, 2):
        gd = jax.grad(lambda a, b: loss("direct", a, b), argnums=argn - 1)(Xb, Yb)
        gg = jax.grad(lambda a, b: loss("gram", a, b), argnums=argn - 1)(Xb, Yb)
        assert _rel(gg, gd) <= 1e-10


def test_chunk_batch_matches():
    rng = np.random.default_rng(4)
    G, n_obs, ens = 23, 15, 4
    grid_ll, obs_ll = _geom(rng, G, n_obs)
    Xb, Yb, Y, rinv = _inputs(rng, G, n_obs, ens)
    idx = jnp.arange(n_obs)
    ref = _make(G, ens, 4, "direct", grid_ll, obs_ll)._localized_analysis(
        Xb, Yb, Y, rinv, idx, rho=1.0)
    for cb in (2, 3, 10):
        r = _make(G, ens, 4, "gram", grid_ll, obs_ll, chunk_batch=cb
                  )._localized_analysis(Xb, Yb, Y, rinv, idx, rho=1.0)
        assert _rel(r, ref) <= 1e-10


def _shapes(jaxpr):
    out = set()
    for eqn in jaxpr.eqns:
        for v in eqn.outvars:
            out.add(tuple(getattr(v.aval, "shape", ())))
        for p in eqn.params.values():
            subs = p if isinstance(p, (list, tuple)) else [p]
            for sp in subs:
                inner = getattr(sp, "jaxpr", sp)
                if hasattr(inner, "eqns"):
                    out |= _shapes(inner)
    return out


def test_gram_jaxpr_has_no_chunk_K_nobs_array():
    rng = np.random.default_rng(0)
    G, n_obs, ens, chunk = 40, 37, 3, 8
    grid_ll, obs_ll = _geom(rng, G, n_obs)
    Xb, Yb, Y, rinv = _inputs(rng, G, n_obs, ens)
    idx = jnp.arange(n_obs)
    bad = {(chunk, ens, n_obs), (chunk, n_obs, ens)}

    def shapes(mode):
        c = _make(G, ens, chunk, mode, grid_ll, obs_ll)
        f = lambda xb: c._localized_analysis(xb, Yb, Y, rinv, idx, rho=1.0)
        return _shapes(jax.make_jaxpr(f)(Xb).jaxpr)

    assert bad & shapes("direct")            # sanity: legacy path has it
    assert not (bad & shapes("gram"))
    c = _make(G, ens, chunk, "auto", grid_ll, obs_ll)
    assert c._use_gram(G, ens) and not c._use_gram(G, chunk + 1)
