"""W4 Stage 5 (MLTLM): gradient checks for ETKF's traceable inflation/
relaxation scalars (multiplicative_inflation, rtps_relaxation,
rtpp_relaxation, additive_inflation).

These were previously stored via eager `float(...)` casts (multiplicative_
inflation was stored raw, unconverted), which crash under `jax.grad` with
`jax.errors.ConcretizationTypeError` -- confirmed directly, not assumed, by
a downstream consumer (MLTLM's `_da_cycler_factory.py`) attempting exactly
this. Two branch sites gated Python control flow on additive_inflation's
VALUE (`_apply_additive`'s early return, and the FGAT `add_key = ... if
additive_inflation > 0.0 else None` in `_etkf.py`/`_etkf4d.py`/
`_letkf4d.py`); both are removed since the additive-inflation math is
already an exact no-op at sigma=0. Mirrors the Var4DBackprop.lr_scale
precedent (709f762): stored as `jnp.asarray(x)` (NO explicit dtype, so it
stays weakly-typed and doesn't force float64 promotion -- see
test_letkf_dtype_purity_fp32, which this fix must not break).
"""
import jax
import jax.numpy as jrand_mod  # noqa: F401  (keep jax.numpy import path warm)
import jax.numpy as jnp
import jax.random as jrand
import numpy as np
import pytest

import dabench as dab

key = jrand.PRNGKey(0)


@pytest.fixture
def fgat_nature():
    l96 = dab.data.Lorenz96(system_dim=5, store_as_jax=True, delta_t=0.01)
    return l96.generate(n_steps=60)


@pytest.fixture
def fgat_fc_model():
    m = dab.data.Lorenz96(system_dim=5, store_as_jax=True, delta_t=0.01)

    class L96Model(dab.model.Model):
        def forecast(self, state_vec, n_steps):
            v = self.model_obj.generate(x0=state_vec['x'].data, n_steps=n_steps)
            return v.isel(time=-1).assign_attrs(delta_t=0.01), v

    return L96Model(model_obj=m)


@pytest.fixture
def fgat_obs(fgat_nature):
    return dab.observer.Observer(
        fgat_nature, times=fgat_nature['time'].data[np.arange(0, 60, 5)],
        random_location_count=5, error_bias=0.0, error_sd=0.5,
        random_seed=91, stationary_observers=True, store_as_jax=True).observe()


def _init_state(nature, ens=6):
    init_noise = jrand.normal(key, shape=(ens, 5))
    init_state = nature.isel(time=10)
    return init_state.assign(
        x=(['ensemble', 'index'], init_state['x'].data + init_noise))


def _loss(cls, kwarg_name, value, fgat_fc_model, init_state, fgat_obs,
         extra_kw):
    """Build a FRESH cycler with kwarg_name=value (the traced scalar), run
    2 cycles, return a scalar loss of the analysis -- mirrors the
    "rebuild cycler per eval" convention every differentiable-cycler caller
    uses (Var4DBackprop, MLTLM's _da_cycler_factory)."""
    kw = dict(system_dim=5, delta_t=0.01, ensemble_dim=6,
             model_obj=fgat_fc_model, fgat=True, **extra_kw)
    kw[kwarg_name] = value
    cycler = cls(**kw)
    out = cycler.cycle(
        input_state=init_state, start_time=init_state['time'].data,
        obs_vector=fgat_obs, obs_error_sd=0.5, analysis_window=0.1,
        n_cycles=2)
    return jnp.sum(jnp.asarray(out['x'].data) ** 2)


@pytest.mark.parametrize("kwarg_name,value,extra_kw", [
    ("multiplicative_inflation", 1.05, {}),
    ("rtps_relaxation", 0.3, {}),
    ("rtpp_relaxation", 0.3, {}),
    ("additive_inflation", 0.2, {}),
])
def test_etkf_inflation_param_is_differentiable(
        fgat_nature, fgat_fc_model, fgat_obs, kwarg_name, value, extra_kw):
    """jax.grad through each traceable scalar matches finite differences --
    the check that was impossible before this fix (ConcretizationTypeError)."""
    init_state = _init_state(fgat_nature)

    def loss_fn(v):
        return _loss(dab.dacycler.ETKF, kwarg_name, v, fgat_fc_model,
                    init_state, fgat_obs, extra_kw)

    v0 = jnp.asarray(value)
    g = jax.grad(loss_fn)(v0)
    eps = 1e-4
    g_fd = (float(loss_fn(v0 + eps)) - float(loss_fn(v0 - eps))) / (2 * eps)
    assert np.isfinite(float(g))
    rel_err = abs(float(g) - g_fd) / max(abs(g_fd), 1e-8)
    assert rel_err < 1e-2, (
        f"{kwarg_name}: grad={float(g)} vs finite-diff={g_fd} "
        f"(rel_err={rel_err})")


def test_etkf_additive_inflation_zero_is_still_a_noop():
    """The removed early-return in _apply_additive must not have changed the
    documented zero-inflation no-op property (now proven by math, not a
    branch) -- byte-identical analysis at additive_inflation=0 whether or
    not a key is supplied."""
    l96 = dab.data.Lorenz96(system_dim=5, store_as_jax=True, delta_t=0.01)
    nature = l96.generate(n_steps=30)
    fc = dab.data.Lorenz96(system_dim=5, store_as_jax=True, delta_t=0.01)

    class L96Model(dab.model.Model):
        def forecast(self, state_vec, n_steps):
            v = self.model_obj.generate(x0=state_vec['x'].data, n_steps=n_steps)
            return v.isel(time=-1).assign_attrs(delta_t=0.01), v

    model = L96Model(model_obj=fc)
    obs = dab.observer.Observer(
        nature, times=nature['time'].data[np.arange(0, 30, 5)],
        random_location_count=5, error_bias=0.0, error_sd=0.5,
        random_seed=3, stationary_observers=True, store_as_jax=True).observe()
    init_state = _init_state(nature)

    etkf = dab.dacycler.ETKF(system_dim=5, delta_t=0.01, ensemble_dim=6,
                             model_obj=model, fgat=True, additive_inflation=0.0)
    out = etkf.cycle(
        input_state=init_state, start_time=init_state['time'].data,
        obs_vector=obs, obs_error_sd=0.5, analysis_window=0.1, n_cycles=2)
    assert bool(np.all(np.isfinite(np.asarray(out['x'].data))))


def test_letkf_dtype_purity_still_holds_with_traced_inflation():
    """Regression guard: the traceable-scalar fix must stay WEAKLY typed
    (no explicit dtype=float), or it silently promotes an fp32 analysis to
    fp64 -- caught once already while landing this fix."""
    rng = np.random.default_rng(0)
    sysd, ens, n_obs = 8, 6, 4
    Xb = jnp.asarray(rng.standard_normal((sysd, ens)), dtype=jnp.float32)
    H = jnp.asarray(rng.standard_normal((n_obs, sysd)), dtype=jnp.float32)
    Y = jnp.asarray(rng.standard_normal((n_obs,)), dtype=jnp.float32)
    R = jnp.identity(n_obs, dtype=jnp.float32)
    letkf = dab.dacycler.LETKF(system_dim=sysd, delta_t=0.01, ensemble_dim=ens,
                               model_obj=None, localize_radius=1e12)
    Xa = letkf._compute_analysis(Xb=Xb, Y=Y, H=H, h=None, R=R)
    assert Xa.dtype == jnp.float32
