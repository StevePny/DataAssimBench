"""Per-time masked H without materialising (T, N, D).

With a user-supplied single-time ``H`` (N, D) and a non-stationary
(NaN-masked) fixed observation pool, the cyclers used to build
``Hs = mask[:, :, None] * H`` / ``jnp.repeat(H, T)`` of shape (T, N, D),
which OOMs for large networks.  :class:`PerTimeH` applies the mask to the
product instead.  These tests check:

  * the analyses are unchanged (old dense Hs vs new lazy Hs), for
    Var4DOperator, Var4DOperatorWC, Var4D and LETKF4D, stationary and
    non-stationary networks;
  * the traced cycle contains no (T, N, D)-shaped intermediate.
"""

import jax
import jax.numpy as jnp
import jax.random as jrand
import numpy as np
import pytest
import xarray as xr

import dabench as dab
from dabench.dacycler import (BFactors, LETKF4D, Var4D, Var4DOperator,
                              Var4DOperatorWC)
import dabench.dacycler._etkf as _etkf_mod
import dabench.dacycler._etkf4d as _etkf4d_mod
import dabench.dacycler._letkf4d as _letkf4d_mod
import dabench.dacycler._var4d as _var4d_mod
import dabench.dacycler._var4d_operator as _var4d_op_mod
import dabench.dacycler._var4d_operator_wc as _var4d_wc_mod
from dabench.dacycler._per_time_h import PerTimeH

jax.config.update("jax_enable_x64", True)

D = 6
N = 9          # pool slots (> D on purpose: shape (T, N, D) is unambiguous)
DT = 0.01
N_CYCLES = 4
OBS_SD = 0.1


def l96_rk4_step(F: float = 8.0, dt: float = DT):
    def f(x):
        return (jnp.roll(x, -1) - jnp.roll(x, 2)) * jnp.roll(x, 1) - x + F

    def step(x):
        k1 = f(x)
        k2 = f(x + 0.5 * dt * k1)
        k3 = f(x + 0.5 * dt * k2)
        k4 = f(x + dt * k3)
        return x + dt / 6.0 * (k1 + 2 * k2 + 2 * k3 + k4)
    return step


class L96RK4Model(dab.model.Model):
    def __init__(self):
        self.step = l96_rk4_step()
        self.model_obj = None

    def forecast(self, state_vec, n_steps):
        x0 = state_vec['x'].data

        def body(x, _):
            xn = jax.vmap(self.step)(x) if x.ndim == 2 else self.step(x)
            return xn, xn
        _, tail = jax.lax.scan(body, x0, None, length=n_steps - 1)
        X = jnp.concatenate([x0[None], tail])
        if x0.ndim == 2:
            ds = xr.Dataset({'x': (('time', 'ensemble', 'index'), X)},
                            coords={'time': np.arange(n_steps)})
            ds = ds.transpose('ensemble', 'time', 'index')
        else:
            ds = xr.Dataset({'x': (('time', 'index'), X)},
                            coords={'time': np.arange(n_steps)})
        return ds.isel(time=-1), ds

    def compute_tlm(self, state_vec, n_steps):
        x0 = state_vec['x'].data
        J = jax.jacfwd(self.step)

        def body(carry, _):
            x, M = carry
            Mn = J(x) @ M
            xn = self.step(x)
            return (xn, Mn), (xn, Mn)
        _, (xs, Ms) = jax.lax.scan(body, (x0, jnp.eye(D)), None,
                                   length=n_steps - 1)
        X = jnp.concatenate([x0[None], xs])
        M = jnp.concatenate([jnp.eye(D)[None], Ms])
        X_ds = xr.Dataset({'x': (('time', 'index'), X)},
                          coords={'time': np.arange(n_steps)})
        return X_ds.assign_attrs(state_vec.attrs), xr.DataArray(M)


def tlm_factory(model, x_linear):
    del x_linear
    return lambda x_t, dx_t: jax.jvp(model.step, (x_t,), (dx_t,))[1]


@pytest.fixture(scope="module")
def nature():
    base = dab.data.Lorenz96(system_dim=D, store_as_jax=True,
                             delta_t=DT).generate(n_steps=80)
    step = l96_rk4_step()
    xs = [base['x'].data[0]]
    for _ in range(79):
        xs.append(step(xs[-1]))
    nat = base.copy()
    nat['x'] = (base['x'].dims, jnp.stack(xs))
    return nat


@pytest.fixture(scope="module")
def H_pool():
    return jrand.normal(jrand.PRNGKey(3), (N, D), dtype=jnp.float64)


def _pool_obs(nature, H, stationary: bool):
    """Fixed-pool obs (N slots) of y = H x + e, NaN-masked if non-stationary."""
    times_idx = np.arange(0, 80, 5)
    T = times_idx.shape[0]
    x = np.asarray(nature['x'].data)[times_idx]                # (T, D)
    rng = np.random.default_rng(7)
    y = x @ np.asarray(H).T + OBS_SD * rng.standard_normal((T, N))
    if not stationary:
        active = rng.random((T, N)) < 0.6
        active[:, 0] = True
        y = np.where(active, y, np.nan)
    sys_idx = np.tile(np.arange(N) % D, (1, T, 1))
    return xr.Dataset(
        {'x': (('time', 'observations'), jnp.asarray(y)),
         'system_index': (('variable', 'time', 'observations'), sys_idx),
         'errors': (('variable', 'time', 'observations'),
                    np.zeros((1, T, N)))},
        coords={'time': np.asarray(nature['time'].data)[times_idx],
                'index': ('observations', np.arange(N) % D),
                'variable': ['x']},
        attrs={'store_as_jax': True, 'system_dim': D, 'delta_t': DT,
               'stationary_observers': stationary, 'error_sd': OBS_SD})


def _old_make_per_time_H(H, n_times, obs_loc_mask, stationary):
    """The pre-fix dense construction (reference)."""
    H = jnp.asarray(H)[jnp.newaxis]
    return jax.lax.cond(
            stationary,
            lambda: jnp.repeat(H, n_times, axis=0),
            lambda: (obs_loc_mask[:, :, jnp.newaxis] * H))


def _old_repeat(H, n_times, mask=None):
    return jnp.repeat(jnp.asarray(H)[jnp.newaxis], n_times, axis=0)


def _var_cycler(cls, H):
    common = dict(system_dim=D, delta_t=DT, model_obj=L96RK4Model(),
                  obs_window_indices=[0, 5, 10], steps_per_window=11, H=H)
    if cls is Var4D:
        return cls(B=jnp.eye(D) * 0.5, n_outer_loops=1, **common)
    if cls is Var4DOperatorWC:
        # Q_half set, else WC delegates to the strong-constraint parent
        common["Q_half"] = 0.1
    return cls(tlm_op_factory=tlm_factory, n_outer_loops=2, n_inner_loops=30,
               B_factors=BFactors(U=jnp.zeros((D, 0)), sigma=jnp.zeros((0,)),
                                  sigma_bg=float(np.sqrt(0.5))),
               **common)


def _run_var(cls, nature, obs, H):
    dc = _var_cycler(cls, H)
    init = nature.isel(time=0) + 0.3 * jrand.normal(jrand.PRNGKey(0), (D,))
    out = dc.cycle(input_state=init, start_time=nature['time'].data[0],
                   obs_vector=obs, obs_error_sd=OBS_SD, n_cycles=N_CYCLES,
                   analysis_window=0.1)
    return np.asarray(out['x'].data)


VAR_MODS = {Var4DOperator: _var4d_op_mod, Var4DOperatorWC: _var4d_wc_mod,
            Var4D: _var4d_mod}


@pytest.mark.parametrize("stationary", [True, False])
@pytest.mark.parametrize("cls", [Var4DOperator, Var4DOperatorWC, Var4D])
def test_var4d_family_matches_dense_Hs(nature, H_pool, cls, stationary,
                                       monkeypatch):
    obs = _pool_obs(nature, H_pool, stationary)
    new = _run_var(cls, nature, obs, H_pool)
    monkeypatch.setattr(VAR_MODS[cls], "make_per_time_H",
                        _old_make_per_time_H)
    old = _run_var(cls, nature, obs, H_pool)
    assert np.all(np.isfinite(new))
    # Var4D (bicgstab tol=1e-5) and WC (larger PCG control space): XLA fusion of
    # the same products perturbs roundoff, which the Krylov iteration
    # amplifies to ~1e-9.  The operator cyclers stay at roundoff.
    tol = 1e-12 if cls is Var4DOperator else 1e-8
    np.testing.assert_allclose(new, old, rtol=tol, atol=tol)
    print(f"{cls.__name__} stationary={stationary} "
          f"max|new-old|={np.max(np.abs(new - old)):.3e}")


def _run_letkf4d(nature, obs, H):
    ens = 8
    init = nature.isel(time=0)
    init = init.assign(x=(['ensemble', 'index'], init['x'].data
                          + jrand.normal(jrand.PRNGKey(1), (ens, D))))
    dc = LETKF4D(system_dim=D, delta_t=DT, ensemble_dim=ens,
                 model_obj=L96RK4Model(), localize_radius=1.5, H=H)
    out = dc.cycle(input_state=init, start_time=nature['time'].data[0],
                   obs_vector=obs, obs_error_sd=OBS_SD, analysis_window=0.1,
                   n_cycles=N_CYCLES)
    return np.asarray(out['x'].data)


@pytest.mark.parametrize("stationary", [True, False])
def test_letkf4d_matches_dense_Hs(nature, H_pool, stationary, monkeypatch):
    obs = _pool_obs(nature, H_pool, stationary)
    new = _run_letkf4d(nature, obs, H_pool)
    monkeypatch.setattr(_etkf_mod, "PerTimeH", _old_repeat)
    monkeypatch.setattr(_etkf4d_mod, "PerTimeH", _old_repeat)
    monkeypatch.setattr(_letkf4d_mod, "PerTimeH", _old_repeat)
    old = _run_letkf4d(nature, obs, H_pool)
    assert np.all(np.isfinite(new))
    np.testing.assert_allclose(new, old, rtol=1e-12, atol=1e-12)
    print(f"LETKF4D stationary={stationary} "
          f"max|new-old|={np.max(np.abs(new - old)):.3e}")


def test_per_time_h_products_and_adjoint():
    T = 4
    H = jrand.normal(jrand.PRNGKey(0), (N, D))
    mask = jrand.bernoulli(jrand.PRNGKey(1), 0.5, (T, N))
    x = jrand.normal(jrand.PRNGKey(2), (D,))
    X = jrand.normal(jrand.PRNGKey(3), (D, 5))
    r = jrand.normal(jrand.PRNGKey(4), (N,))
    lazy = PerTimeH(H, T, mask)
    dense = mask[:, :, None] * H
    np.testing.assert_array_equal(lazy.dense(), dense)
    for t in range(T):
        np.testing.assert_allclose(lazy[t] @ x, dense[t] @ x, atol=1e-13)
        np.testing.assert_allclose(lazy[t] @ X, dense[t] @ X, atol=1e-13)
        g = jax.grad(lambda z: jnp.dot(r, lazy[t] @ z))(x)
        np.testing.assert_allclose(g, dense[t].T @ r, atol=1e-13)
        # masked-out obs contribute exactly zero
        assert np.all(np.asarray(lazy[t] @ x)[~np.asarray(mask[t])] == 0)


def _shapes_in_jaxpr(jaxpr, acc):
    for eqn in jaxpr.eqns:
        for v in list(eqn.outvars) + list(eqn.invars):
            if hasattr(v, "aval") and hasattr(v.aval, "shape"):
                acc.add(tuple(v.aval.shape))
        for p in eqn.params.values():
            subs = p if isinstance(p, (list, tuple)) else [p]
            for s in subs:
                if hasattr(s, "jaxpr"):
                    _shapes_in_jaxpr(getattr(s.jaxpr, "jaxpr", s.jaxpr), acc)
                elif hasattr(s, "eqns"):
                    _shapes_in_jaxpr(s, acc)
    return acc


@pytest.mark.parametrize("cls", [Var4DOperator, LETKF4D])
def test_no_TND_intermediate(nature, H_pool, cls, monkeypatch):
    """Trace one analysis window; no (T, N, D) array may appear."""
    obs = _pool_obs(nature, H_pool, stationary=False)
    T = 3                                    # obs times per window
    if cls is Var4DOperator:
        dc = _var_cycler(cls, H_pool)
        dc._obs_vector = obs.fillna(0)
        dc.obs_error_sd = OBS_SD
        dc._return_metrics = False
        x0 = nature.isel(time=0)
        mask = jnp.asarray(~np.isnan(np.asarray(obs['x'].data[:T])))

        def f(xb, yv, m):
            return dc._cycle_obsop(
                    x0.assign(x=(x0['x'].dims, xb)), yv,
                    jnp.zeros((T, N), int), jnp.ones(T, bool), m,
                    obs_window_indices=[0, 5, 10])['x'].data
        args = (x0['x'].data, jnp.zeros((T, N)), mask)
    else:
        def f(H, mask, x):
            Hs = PerTimeH(H, T, mask)
            return jax.vmap(lambda i: Hs[i] @ x)(jnp.arange(T))
        args = (H_pool, jnp.ones((T, N), bool), jnp.ones(D))
        # LETKF4D path: same object; check the dense reference DOES trip it
        dense_shapes = _shapes_in_jaxpr(jax.make_jaxpr(
            lambda H, m, x: jax.vmap(
                lambda i: _old_repeat(H, T)[i] @ x)(jnp.arange(T)))(
                *args).jaxpr, set())
        assert (T, N, D) in dense_shapes
    shapes = _shapes_in_jaxpr(jax.make_jaxpr(f)(*args).jaxpr, set())
    assert (T, N, D) not in shapes, sorted(shapes)
    if cls is Var4DOperator:
        # sanity: the detector does catch the pre-fix construction
        monkeypatch.setattr(_var4d_op_mod, "make_per_time_H",
                            _old_make_per_time_H)
        # fresh callable: make_jaxpr caches traces per function
        old_shapes = _shapes_in_jaxpr(
            jax.make_jaxpr(lambda *a: f(*a))(*args).jaxpr, set())
        assert (T, N, D) in old_shapes
