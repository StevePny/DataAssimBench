"""Tests for weak-constraint (model-error forcing) 4D-Var cyclers.

Covers :class:`Var4DOperatorWC`, :class:`Var4DBackpropWC` and the helpers
in :mod:`dabench.dacycler._var4d_weak_utils`:

  * no ``Q_half`` -> byte-identical to the strong-constraint parents
    (and to the parents' stored regression values);
  * ``Q_half -> 0`` -> weak-constraint analysis converges to strong;
  * twin experiment with a biased forecast model (Lorenz96, wrong F):
    weak constraint beats strong constraint on analysis error;
  * incremental (GN) gradient of the weak-constraint cost equals
    ``jax.grad`` of the full nonlinear cost (and a finite difference);
  * the accepted ``Q_half`` forms agree.

The cycling tests use a pure-JAX RK4 Lorenz96 (odeint's custom rules do
not support the forward-over-reverse HVP the operator inner loop needs).
"""

import jax
import jax.numpy as jnp
import jax.random as jrand
import numpy as np
import pytest
import xarray as xr

import dabench as dab
from dabench.dacycler import BFactors
from dabench.dacycler._var4d_weak_utils import (
        build_Q_half,
        expand_eta,
        forced_rollout,
        quadratic_cost_wc,
        )


D = 6
DT = 0.01
F_TRUE = 8.0
F_BIASED = 11.0          # forecast-model forcing error -> tendency bias
N_CYCLES = 20
key = jrand.PRNGKey(42)


def l96_rk4_step(F: float, dt: float = DT):
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
    """Pure-JAX RK4 Lorenz96 forecast model (forecast frames incl. x0)."""

    def __init__(self, F: float):
        self.step = l96_rk4_step(F)
        self.model_obj = None

    def forecast(self, state_vec, n_steps):
        x0 = state_vec['x'].data

        def body(x, _):
            xn = self.step(x)
            return xn, xn
        _, tail = jax.lax.scan(body, x0, None, length=n_steps - 1)
        X = jnp.concatenate([x0[None], tail])
        ds = xr.Dataset({'x': (('time', 'index'), X)},
                        coords={'time': np.arange(n_steps)})
        return ds.isel(time=-1), ds


def tlm_factory(model, x_linear):
    del x_linear
    return lambda x_t, dx_t: jax.jvp(model.step, (x_t,), (dx_t,))[1]


@pytest.fixture(scope="module")
def nature():
    base = dab.data.Lorenz96(system_dim=D, store_as_jax=True,
                             delta_t=DT).generate(n_steps=220)
    step = l96_rk4_step(F_TRUE)
    xs = [base['x'].data[0]]
    for _ in range(219):
        xs.append(step(xs[-1]))
    nat = base.copy()
    nat['x'] = (base['x'].dims, jnp.stack(xs))
    return nat


@pytest.fixture(scope="module")
def obs(nature):
    return dab.observer.Observer(
        nature, times=nature['time'].data[jnp.arange(0, 220, 5)],
        random_location_count=6, error_bias=0.0, error_sd=0.1,
        random_seed=94, stationary_observers=True,
        store_as_jax=True).observe()


B_FACTORS = BFactors(U=jnp.zeros((D, 0)), sigma=jnp.zeros((0,)),
                     sigma_bg=float(np.sqrt(0.5)))


def _op(cls, F, **kw):
    kw.setdefault("n_outer_loops", 1)
    return cls(system_dim=D, delta_t=DT, model_obj=L96RK4Model(F),
               tlm_op_factory=tlm_factory, B_factors=B_FACTORS,
               obs_window_indices=[0, 5, 10], steps_per_window=11,
               n_inner_loops=30, **kw)


def _bp(cls, F, **kw):
    return cls(system_dim=D, delta_t=DT, model_obj=L96RK4Model(F),
               B=jnp.eye(D) * 0.5, learning_rate=0.1, lr_decay=1.0,
               num_iters=20, obs_window_indices=[0, 5, 10],
               steps_per_window=11, **kw)


def _cycle(dc, nature, obs, **kw):
    init = nature.isel(time=0) + 0.5 * jrand.normal(key, (D,))
    return dc.cycle(input_state=init, start_time=nature['time'].data[0],
                    obs_vector=obs, obs_error_sd=0.1, n_cycles=N_CYCLES,
                    analysis_window=0.1, **kw)


def _ana_rmse(out, nature):
    xa = np.asarray(out['x'].data)
    xt = np.asarray(nature['x'].data)[np.arange(0, 10 * N_CYCLES, 10)]
    return float(np.sqrt(((xa - xt) ** 2).mean(-1))[5:].mean())


# ---------------------------------------------------------------- no-Q path

def test_operator_wc_noQ_identical(nature, obs):
    for n_outer in (1, 3):
        a = _cycle(_op(dab.dacycler.Var4DOperator, F_TRUE,
                       n_outer_loops=n_outer), nature, obs)
        b = _cycle(_op(dab.dacycler.Var4DOperatorWC, F_TRUE,
                       n_outer_loops=n_outer), nature, obs)
        assert np.array_equal(np.asarray(a['x'].data),
                              np.asarray(b['x'].data))
    am, mm = _cycle(_op(dab.dacycler.Var4DOperator, F_TRUE), nature, obs,
                    return_metrics=True)
    bm, wm = _cycle(_op(dab.dacycler.Var4DOperatorWC, F_TRUE), nature, obs,
                    return_metrics=True)
    assert np.array_equal(np.asarray(am['x'].data), np.asarray(bm['x'].data))
    for v in mm.keys():
        assert np.array_equal(np.asarray(mm[v].data), np.asarray(wm[v].data),
                              equal_nan=True)


def test_backprop_wc_noQ_identical_and_stored():
    """Same fixture as tests/dacycler_var4d_var4dbp_test.py (odeint L96)."""
    l96 = dab.data.Lorenz96(system_dim=6, store_as_jax=True, delta_t=0.01)
    nat = l96.generate(n_steps=120)
    ob = dab.observer.Observer(
        nat, times=nat['time'].data[jnp.arange(0, 120, 5)],
        random_location_count=3, error_bias=0.0, error_sd=0.3,
        random_seed=94, stationary_observers=True, store_as_jax=True
        ).observe()
    model_l96 = dab.data.Lorenz96(system_dim=6, store_as_jax=True,
                                  delta_t=0.01)

    class L96Model(dab.model.Model):
        def forecast(self, state_vec, n_steps):
            new_vec = self.model_obj.generate(x0=state_vec['x'].data,
                                              n_steps=n_steps)
            return new_vec.isel(time=-1), new_vec

    outs = []
    for cls in (dab.dacycler.Var4DBackprop, dab.dacycler.Var4DBackpropWC):
        dc = cls(system_dim=6, delta_t=0.01,
                 model_obj=L96Model(model_obj=model_l96),
                 obs_window_indices=[0, 5, 10], steps_per_window=11,
                 learning_rate=0.1, lr_decay=0.5, B=jnp.identity(6) * 0.05)
        out = dc.cycle(input_state=nat.isel(time=0)
                       + jrand.normal(key, shape=(6,)),
                       start_time=nat['time'].data[0], obs_vector=ob,
                       obs_error_sd=ob.error_sd * 1.5, n_cycles=10,
                       analysis_window=0.1, return_forecast=True)
        outs.append(np.asarray(out.stack(
            time=['cycle', 'cycle_timestep']).transpose('time', ...)
            ['x'].values))
    assert np.array_equal(outs[0], outs[1])
    assert jnp.allclose(outs[1][0], jnp.array(
        [4.13798625, 8.12037255, 3.97236149, 1.81120073, 0.18632274,
         0.51606187]))
    assert jnp.allclose(outs[1][-1], jnp.array(
        [1.55696816e-04, 1.41966997e+00, 5.77068664e+00, 5.19543709e+00,
         1.46699910e-01, -4.62494618e+00]))


# ------------------------------------------------------------- Q -> 0 limit

@pytest.mark.parametrize("eta_mode", ["constant", "per_step"])
@pytest.mark.parametrize("n_outer", [1, 3])
def test_operator_wc_tiny_Q_converges_to_strong(nature, obs, eta_mode,
                                                n_outer):
    a = _cycle(_op(dab.dacycler.Var4DOperator, F_BIASED,
                   n_outer_loops=n_outer), nature, obs)
    b = _cycle(_op(dab.dacycler.Var4DOperatorWC, F_BIASED, Q_half=1e-9,
                   eta_mode=eta_mode, n_outer_loops=n_outer), nature, obs)
    assert np.allclose(np.asarray(a['x'].data), np.asarray(b['x'].data),
                       atol=1e-6)


@pytest.mark.parametrize("eta_mode", ["constant", "per_step"])
def test_backprop_wc_tiny_Q_converges_to_strong(nature, obs, eta_mode):
    a = _cycle(_bp(dab.dacycler.Var4DBackprop, F_BIASED), nature, obs)
    b = _cycle(_bp(dab.dacycler.Var4DBackpropWC, F_BIASED, Q_half=1e-9,
                   eta_mode=eta_mode), nature, obs)
    assert np.allclose(np.asarray(a['x'].data), np.asarray(b['x'].data),
                       atol=1e-6)


# ------------------------------------------------------ biased-model twin

@pytest.mark.parametrize("eta_mode,sigma_q", [("constant", 0.02),
                                              ("per_step", 0.05)])
def test_operator_wc_beats_strong_with_biased_model(nature, obs, eta_mode,
                                                    sigma_q):
    e_s = _ana_rmse(_cycle(_op(dab.dacycler.Var4DOperator, F_BIASED),
                           nature, obs), nature)
    e_w = _ana_rmse(_cycle(_op(dab.dacycler.Var4DOperatorWC, F_BIASED,
                               Q_half=sigma_q, eta_mode=eta_mode),
                           nature, obs), nature)
    assert e_w < 0.75 * e_s, (e_w, e_s)


@pytest.mark.parametrize("eta_mode,sigma_q", [("constant", 0.02),
                                              ("per_step", 0.05)])
def test_backprop_wc_beats_strong_with_biased_model(nature, obs, eta_mode,
                                                    sigma_q):
    e_s = _ana_rmse(_cycle(_bp(dab.dacycler.Var4DBackprop, F_BIASED),
                           nature, obs), nature)
    out, metrics = _cycle(_bp(dab.dacycler.Var4DBackpropWC, F_BIASED,
                              Q_half=sigma_q, eta_mode=eta_mode),
                          nature, obs, return_metrics=True)
    e_w = _ana_rmse(out, nature)
    assert e_w < 0.75 * e_s, (e_w, e_s)
    assert bool(np.all(np.isfinite(np.asarray(metrics["o_minus_a_rms"]))))


def test_operator_wc_metrics(nature, obs):
    out, m = _cycle(_op(dab.dacycler.Var4DOperatorWC, F_BIASED, Q_half=0.02,
                        n_outer_loops=2), nature, obs, return_metrics=True)
    of = np.asarray(m["o_minus_f_rms"].data)
    oa = np.asarray(m["o_minus_a_rms"].data)
    assert of.shape == (N_CYCLES,)
    assert bool(np.all(oa <= of + 1e-6))


# ------------------------------------------------- incremental gradient

@pytest.mark.parametrize("eta_mode", ["constant", "per_step"])
def test_incremental_gradient_matches_full_cost(eta_mode):
    """grad of the GN quadratic model at delta=0 == grad of the full
    nonlinear weak-constraint cost at the linearisation point."""
    T, n_obs = 6, 3
    step = l96_rk4_step(9.0, dt=0.05)
    k = jrand.split(jrand.PRNGKey(3), 8)
    x_b = 3.0 * jrand.normal(k[0], (D,))
    B_half = 0.5 * jnp.eye(D) + 0.1 * jrand.normal(k[1], (D, D))
    Q_half = jnp.abs(0.2 + 0.05 * jrand.normal(k[2], (D,)))
    apply_B_half = lambda v: B_half @ v                    # noqa: E731
    apply_Q_half = build_Q_half(Q_half, D)
    n_e = 1 if eta_mode == "constant" else T
    Hs = jrand.normal(k[3], (n_obs, 4, D))
    owi = jnp.array([1, 3, 6])
    y = jrand.normal(k[4], (n_obs, 4))
    mask = jnp.array([True, True, True])
    R_inv = jnp.full((4,), 2.0)
    v_tot = 0.3 * jrand.normal(k[5], (D,))
    chi_tot = 0.3 * jrand.normal(k[6], (n_e, D))

    def traj(v, chi):
        eta = expand_eta(jax.vmap(apply_Q_half)(chi), T)
        return forced_rollout(step, x_b + apply_B_half(v), eta)

    def J_full(v, chi):
        X = traj(v, chi)
        r = jax.vmap(lambda i: Hs[i] @ X[owi[i]] - y[i])(jnp.arange(n_obs))
        return (0.5 * jnp.sum(v * v) + 0.5 * jnp.sum(chi * chi)
                + 0.5 * jnp.sum(R_inv * r * r))

    x_l = traj(v_tot, chi_tot)
    innov = jax.vmap(lambda i: y[i] - Hs[i] @ x_l[owi[i]])(jnp.arange(n_obs))
    tlm = lambda x, dx: jax.jvp(step, (x,), (dx,))[1]     # noqa: E731

    def J_inc(dv, dchi):
        return quadratic_cost_wc(
                dv, dchi, v_tot, chi_tot, tlm_op=tlm, x_l_traj=x_l, Hs=Hs,
                innovations=innov, obs_window_indices=owi,
                obs_time_mask=mask, R_inv_diag=R_inv,
                apply_B_half=apply_B_half, apply_Q_half=apply_Q_half)

    g_inc = jax.grad(J_inc, argnums=(0, 1))(jnp.zeros(D),
                                             jnp.zeros((n_e, D)))
    g_full = jax.grad(J_full, argnums=(0, 1))(v_tot, chi_tot)
    assert jnp.allclose(g_inc[0], g_full[0], rtol=1e-8, atol=1e-8)
    assert jnp.allclose(g_inc[1], g_full[1], rtol=1e-8, atol=1e-8)
    # Value at delta=0 equals the full cost; FD directional check.
    assert jnp.allclose(J_inc(jnp.zeros(D), jnp.zeros((n_e, D))),
                        J_full(v_tot, chi_tot), rtol=1e-10)
    pv = jrand.normal(k[7], (D,))
    pc = jrand.normal(k[7], (n_e, D))
    h = 1e-6
    fd = (J_full(v_tot + h * pv, chi_tot + h * pc)
          - J_full(v_tot - h * pv, chi_tot - h * pc)) / (2 * h)
    assert jnp.allclose(fd, jnp.vdot(g_inc[0], pv) + jnp.vdot(g_inc[1], pc),
                        rtol=1e-5)


# ------------------------------------------------------------ Q_half forms

def test_build_Q_half_forms():
    chi = jrand.normal(jrand.PRNGKey(5), (D,))
    q = jnp.linspace(0.1, 0.6, D)
    assert build_Q_half(None, D) is None
    assert jnp.allclose(build_Q_half(0.3, D)(chi), 0.3 * chi)
    assert jnp.allclose(build_Q_half(q, D)(chi), q * chi)
    assert jnp.allclose(build_Q_half(jnp.diag(q), D)(chi), q * chi)
    assert jnp.allclose(build_Q_half(lambda c: q * c, D)(chi), q * chi)
    bf = BFactors(U=jnp.zeros((D, 0)), sigma=jnp.zeros((0,)), sigma_bg=0.3)
    assert jnp.allclose(build_Q_half(bf, D)(chi), 0.3 * chi)
    with pytest.raises(ValueError):
        build_Q_half(jnp.ones(D + 1), D)
    with pytest.raises(ValueError):
        dab.dacycler.Var4DOperatorWC(
                system_dim=D, delta_t=DT, model_obj=L96RK4Model(F_TRUE),
                tlm_op_factory=tlm_factory, Q_half=0.1, eta_mode="bogus")
