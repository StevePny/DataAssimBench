"""Compiled-window path of the matrix-free 4D-Var cyclers.

``cycle(model_inputs=..., obs_inputs=..., compile_once=True)``: the model time
and per-step inputs reach the step ``F(x, t, u)`` as data, a time-indexed
linear observation operator ``H(x, o_i)`` takes per-time coefficients, and the
window is one jitted function reused across cycles, calls and (through JAX's
persistent compilation cache) processes.

  * the new path equals the original one with the inputs carried by a "clock"
    element in the state (strong and weak constraint);
  * a second ``cycle()`` call with different data compiles nothing;
  * a new cycler instance with an equal configuration in a new process hits
    the persistent compilation cache;
  * VarBC twin: per-pass bias and tilt as state elements observed through
    ``H(x, o_i)`` are recovered;
  * the other cyclers refuse the new arguments.
"""

import os
import subprocess
import sys
import textwrap

import jax
import jax.numpy as jnp
import numpy as np
import pytest
import xarray as xr

import dabench as dab
from dabench.dacycler._per_time_h import LinearObsOperator

jax.config.update("jax_enable_x64", True)

N = 12          # physical state size
S = 6           # frames per window (5 model steps)
DT = 1.0
N_CYCLES = 3
N_FRAMES = N_CYCLES * (S - 1) + S
SIG_B = 0.3
SIG_Q = 0.02


def tendency(x):
    return -0.1 * x + 0.3 * jnp.sin(jnp.roll(x, 1)) - 0.1 * jnp.tanh(jnp.roll(x, -1) * x)


def phys_step(x, f):
    return x + 0.25 * (tendency(x) + f)


def frames(seed=0):
    rng = np.random.default_rng(seed)
    base = np.sin(np.linspace(0, 2 * np.pi, N))[None] * np.cos(0.3 * np.arange(N_FRAMES))[:, None]
    return base + 0.1 * rng.standard_normal((N_FRAMES, N))


class ClockModel(dab.model.Model):
    """State [x | clock]; the step reads its forcing frame by the clock (the
    workaround the time argument replaces)."""

    def __init__(self, f):
        self.f = jnp.asarray(f)
        self.model_obj = None
        self.delta_t = DT

    def step(self, z):
        x, clock = z[:N], z[N]
        k = jnp.clip(clock.astype(int), 0, self.f.shape[0] - 1)
        return jnp.concatenate([phys_step(x, self.f[k]), clock[None] + 1.0])

    def forecast(self, state_vec, n_steps):
        z0 = jnp.asarray(state_vec['x'].data)

        def body(z, _):
            zn = self.step(z)
            return zn, zn
        _, tail = jax.lax.scan(body, z0, None, length=n_steps - 1)
        Z = jnp.concatenate([z0[None], tail])
        ds = xr.Dataset({'x': (('time', 'index'), Z)}, coords={'time': np.arange(n_steps) * DT})
        return ds.isel(time=-1, drop=True), ds


class InputsModel(dab.model.Model):
    """State x; the forcing frame arrives as the step's input u."""

    def __init__(self):
        self.model_obj = None
        self.delta_t = DT

    def step_fn(self):
        return lambda x, t, u: phys_step(x, u)

    def forecast(self, state_vec, n_steps):
        raise NotImplementedError("only the compiled-window path is used")


def clock_tlm_factory(model, x_linear):
    return lambda z_t, dz_t: jax.jvp(model.step, (z_t,), (dz_t,))[1]


def nature(f, x0):
    xs = [x0]
    for k in range(N_FRAMES - 1):
        xs.append(np.asarray(phys_step(jnp.asarray(xs[-1]), f[k])))
    return np.stack(xs)


def obs_vector(truth, locs, seed=1, sd=0.05, times=None):
    rng = np.random.default_rng(seed)
    times = np.arange(0, N_CYCLES * (S - 1) + 1) if times is None else times
    vals = truth[times][:, locs] + sd * rng.standard_normal((len(times), len(locs)))
    return xr.Dataset(
        {'x': (('time', 'observations'), vals),
         'system_index': (('variable', 'time', 'observations'),
                          np.broadcast_to(np.asarray(locs), (1, len(times), len(locs))).copy())},
        coords={'time': times.astype(float), 'observations': np.arange(len(locs)), 'variable': ['x']},
        attrs=dict(stationary_observers=True, error_sd=sd))


LOCS = [0, 2, 5, 7, 9, 11]


def state(x):
    return xr.Dataset({'x': (('index',), np.asarray(x, float))}, coords={'index': np.arange(len(x)), 'time': 0.0})


def _setup(seed=0):
    f = frames(seed)
    x0 = np.cos(np.linspace(0, 2 * np.pi, N))
    truth = nature(f, x0)
    xb = x0 + 0.3 * np.random.default_rng(7).standard_normal(N)
    return f, truth, xb


def _clock_cycler(cls, f, **kw):
    B = lambda v: SIG_B * v * jnp.concatenate([jnp.ones(N), jnp.zeros(1)])
    if 'Q_half' in kw:
        kw['Q_half'] = lambda chi: SIG_Q * chi * jnp.concatenate([jnp.ones(N), jnp.zeros(1)])
    return cls(system_dim=N + 1, delta_t=DT, model_obj=ClockModel(f), tlm_op_factory=clock_tlm_factory,
               B_half_op=B, n_outer_loops=2, n_inner_loops=40, inner_tol=1e-12, steps_per_window=S, **kw)


def _inputs_cycler(cls, **kw):
    if 'Q_half' in kw:
        kw['Q_half'] = SIG_Q
    return cls(system_dim=N, delta_t=DT, model_obj=InputsModel(), tlm_op_factory=None,
               B_half_op=lambda v: SIG_B * v, n_outer_loops=2, n_inner_loops=40, inner_tol=1e-12,
               steps_per_window=S, **kw)


@pytest.mark.parametrize("cls,kw", [(dab.dacycler.Var4DOperator, {}),
                                    (dab.dacycler.Var4DOperatorWC, dict(Q_half=True, eta_mode="constant")),
                                    (dab.dacycler.Var4DOperatorWC, dict(Q_half=True, eta_mode="per_step"))])
def test_inputs_path_equals_clock_path(cls, kw):
    f, truth, xb = _setup()
    obs = obs_vector(truth, LOCS)
    common = dict(start_time=0.0, obs_vector=obs, obs_error_sd=0.05, n_cycles=N_CYCLES,
                  analysis_window=float(S - 1), return_forecast=True, return_background=True)
    old = _clock_cycler(cls, f, **kw).cycle(
        input_state=state(np.concatenate([xb, [0.0]])), **common)
    new = _inputs_cycler(cls, **kw).cycle(
        input_state=state(xb), model_inputs=jnp.asarray(f), **common)
    a_old, a_new = np.asarray(old[0]['x'].data)[..., :N], np.asarray(new[0]['x'].data)
    b_old, b_new = np.asarray(old[1]['x'].data)[..., :N], np.asarray(new[1]['x'].data)
    assert a_old.shape == a_new.shape == (N_CYCLES, S - 1, N)
    np.testing.assert_allclose(a_new, a_old, rtol=1e-9, atol=1e-11)
    np.testing.assert_allclose(b_new, b_old, rtol=1e-9, atol=1e-11)
    # the analysis moved towards the truth (the comparison is not vacuous)
    assert np.abs(a_new[:, 0] - truth[0:N_CYCLES * (S - 1):S - 1]).mean() < np.abs(xb - truth[0]).mean()


def _count_backend_compiles():
    n = [0]

    def cb(event, duration, **kw):
        if event == '/jax/core/compile/backend_compile_duration':
            n[0] += 1
    jax.monitoring.register_event_duration_secs_listener(cb)
    return n


@pytest.mark.parametrize("cls,kw", [(dab.dacycler.Var4DOperator, {}),
                                    (dab.dacycler.Var4DOperatorWC, dict(Q_half=True, eta_mode="constant"))])
def test_second_call_with_new_data_compiles_nothing(cls, kw):
    f1, truth1, xb1 = _setup(0)
    f2, truth2, xb2 = _setup(3)
    dc = _inputs_cycler(cls, **kw)
    common = dict(start_time=0.0, obs_error_sd=0.05, n_cycles=N_CYCLES, analysis_window=float(S - 1),
                  return_forecast=True)
    n = _count_backend_compiles()
    try:
        r1 = dc.cycle(input_state=state(xb1), obs_vector=obs_vector(truth1, LOCS, seed=1),
                      model_inputs=jnp.asarray(f1), **common)
        n1 = n[0]
        r2 = dc.cycle(input_state=state(xb2), obs_vector=obs_vector(truth2, LOCS, seed=2),
                      model_inputs=jnp.asarray(f2), **common)
        n2 = n[0] - n1
    finally:
        jax.monitoring.clear_event_listeners()
    assert n1 >= 1
    assert n2 == 0, f"{n2} compilations on the second call with new data"
    assert not np.allclose(np.asarray(r1['x'].data), np.asarray(r2['x'].data))


def test_persistent_cache_across_instances_and_processes(tmp_path):
    """A fresh process builds a new cycler (new closures, new model object) with
    an equal configuration: the window comes from the persistent cache."""
    script = textwrap.dedent(f"""
        import sys, jax, jax.numpy as jnp, numpy as np, xarray as xr
        jax.config.update("jax_compilation_cache_dir", {str(tmp_path / 'cache')!r})
        jax.config.update("jax_persistent_cache_min_compile_time_secs", 0)
        jax.config.update("jax_persistent_cache_min_entry_size_bytes", -1)
        sys.path.insert(0, {os.path.dirname(__file__)!r})
        import dacycler_compiled_window_test as t
        hits, misses = [0], [0]
        def cb(e, **k):
            hits[0] += e == '/jax/compilation_cache/cache_hits'
            misses[0] += e == '/jax/compilation_cache/cache_misses'
        jax.monitoring.register_event_listener(cb)
        seed = int(sys.argv[1])
        f, truth, xb = t._setup(seed)
        dc = t._inputs_cycler(t.dab.dacycler.Var4DOperator)
        dc.cycle(input_state=t.state(xb), start_time=0.0,
                 obs_vector=t.obs_vector(truth, t.LOCS, seed=seed + 1), obs_error_sd=0.05,
                 n_cycles=t.N_CYCLES, analysis_window=float(t.S - 1), model_inputs=jnp.asarray(f))
        print(hits[0], misses[0])
    """)
    path = tmp_path / "run.py"
    path.write_text(script)
    env = dict(os.environ, PYTHONPATH=os.pathsep.join([os.path.dirname(os.path.dirname(dab.__file__)),
                                                       os.environ.get("PYTHONPATH", "")]))
    out = [subprocess.run([sys.executable, str(path), str(s)], capture_output=True, text=True, env=env, check=True)
           .stdout.split() for s in (0, 3)]
    (h1, m1), (h2, m2) = [(int(a), int(b)) for a, b in out]
    assert m1 >= 1                       # first process compiles
    assert m2 == 0 and h2 >= 1, (h1, m1, h2, m2)   # second: every executable from the cache


def test_varbc_twin_recovers_pass_bias_and_tilt():
    """State [x | a_p | b_p] (P passes, constant in time); H(z, o_i) = z[loc] + a[p] + b[p] s with o_i = (p, s)
    per observation location and time.  Pass 0 is the reference (a = b = 0, no background spread)."""
    P = 3
    rng = np.random.default_rng(11)
    f, truth, _ = _setup()
    a_true, b_true = np.array([0.0, 0.12, -0.08]), np.array([0.0, 0.05, 0.10])
    nt = S                                  # one window, observations at every frame
    times = np.arange(nt)
    pid = np.stack([rng.permutation(np.repeat(np.arange(P), 2)) for _ in times])
    s_km = rng.uniform(-1.0, 1.0, (nt, len(LOCS)))
    locs = np.asarray(LOCS)
    y = truth[times][:, locs] + a_true[pid] + b_true[pid] * s_km + 0.01 * rng.standard_normal(pid.shape)
    obs = obs_vector(truth, LOCS, times=times)
    obs['x'] = (('time', 'observations'), y)
    obs.attrs['error_sd'] = 0.01

    def H_fn(z, o):
        p, s = o
        return z[locs] + z[N + p] + z[N + P + p] * s
    H = LinearObsOperator(H_fn, (len(LOCS), N + 2 * P), takes_inputs=True)

    class BiasModel(InputsModel):
        def step_fn(self):
            return lambda z, t, u: jnp.concatenate([phys_step(z[:N], u), z[N:]])

    sig = np.concatenate([np.full(N, 0.02), [0.0, 0.3, 0.3], [0.0, 0.3, 0.3]])
    dc = dab.dacycler.Var4DOperator(system_dim=N + 2 * P, delta_t=DT, model_obj=BiasModel(), tlm_op_factory=None,
                                    B_half_op=lambda v: jnp.asarray(sig) * v, H=H, n_outer_loops=2,
                                    n_inner_loops=80, inner_tol=1e-12, steps_per_window=S)
    xb = np.concatenate([truth[0] + 0.02 * rng.standard_normal(N), np.zeros(2 * P)])
    out = dc.cycle(input_state=state(xb), start_time=0.0, obs_vector=obs,
                   obs_error_sd=0.01, n_cycles=1, analysis_window=float(S - 1), model_inputs=jnp.asarray(f),
                   obs_inputs=(pid, s_km))
    za = np.asarray(out['x'].data)[0]
    a_est, b_est = za[N:N + P], za[N + P:]
    np.testing.assert_allclose(a_est, a_true, atol=0.01)
    np.testing.assert_allclose(b_est, b_true, atol=0.015)


def test_other_cyclers_refuse_new_arguments():
    dc = dab.dacycler.Var3D(system_dim=N, delta_t=DT, model_obj=InputsModel())
    with pytest.raises(NotImplementedError):
        dc.cycle(input_state=xr.Dataset({'x': (('index',), np.zeros(N))}), start_time=0.0,
                 obs_vector=obs_vector(np.zeros((N_FRAMES, N)), LOCS), n_cycles=1, model_inputs=np.zeros((5, N)))


def test_metrics_refused_on_compiled_path():
    f, truth, xb = _setup()
    with pytest.raises(NotImplementedError):
        _inputs_cycler(dab.dacycler.Var4DOperator).cycle(
            input_state=state(xb), start_time=0.0, obs_vector=obs_vector(truth, LOCS),
            n_cycles=1, analysis_window=float(S - 1), model_inputs=jnp.asarray(f), return_metrics=True)


# --- an observation error paired with each observation value ---

PER_OBS_CASES = [(dab.dacycler.Var4DOperator, {}),
                 (dab.dacycler.Var4DOperatorWC, dict(Q_half=True, eta_mode="constant"))]


def _per_obs_run(cls, kw, sd, **extra):
    f, truth, xb = _setup()
    obs = obs_vector(truth, LOCS)
    return np.asarray(_inputs_cycler(cls, **kw).cycle(
        input_state=state(xb), obs_vector=obs, obs_error_sd=sd, start_time=0.0, n_cycles=N_CYCLES,
        analysis_window=float(S - 1), return_forecast=True, model_inputs=jnp.asarray(f), **extra)['x'].data), obs


@pytest.mark.parametrize("cls,kw", PER_OBS_CASES)
def test_per_observation_error_repeating_a_vector_equals_the_vector(cls, kw):
    vec = np.array([0.03, 0.05, 0.05, 0.08, 0.2, 0.05])
    a_vec, obs = _per_obs_run(cls, kw, vec)
    nt = obs['x'].shape[0]
    a_arr, _ = _per_obs_run(cls, kw, np.tile(vec, (nt, 1)))
    np.testing.assert_allclose(a_arr, a_vec, rtol=1e-9, atol=1e-11)


@pytest.mark.parametrize("cls,kw", PER_OBS_CASES)
def test_per_observation_error_is_per_time_and_nan_means_no_weight(cls, kw):
    _, obs = _per_obs_run(cls, kw, 0.05)
    nt, no = obs['x'].shape
    sd = np.full((nt, no), 0.05)
    sd[nt // 2:, :3] = 1.0e6                      # late observations of three elements: practically no weight
    a_big, _ = _per_obs_run(cls, kw, sd)
    sd_nan = np.where(sd > 1.0e5, np.nan, sd)
    a_nan, _ = _per_obs_run(cls, kw, sd_nan)
    np.testing.assert_allclose(a_nan, a_big, rtol=1e-6, atol=1e-8)
    a_uniform, _ = _per_obs_run(cls, kw, 0.05)
    assert np.abs(a_nan - a_uniform).max() > 1e-4                      # the per-time errors matter


def test_per_observation_error_needs_the_compiled_path_and_the_right_shape():
    f, truth, xb = _setup()
    obs = obs_vector(truth, LOCS)
    nt, no = obs['x'].shape
    with pytest.raises(NotImplementedError):
        _clock_cycler(dab.dacycler.Var4DOperator, f).cycle(
            input_state=state(np.concatenate([xb, [0.0]])), obs_vector=obs, obs_error_sd=np.full((nt, no), 0.05),
            start_time=0.0, n_cycles=N_CYCLES, analysis_window=float(S - 1))
    with pytest.raises(ValueError):
        _inputs_cycler(dab.dacycler.Var4DOperator).cycle(
            input_state=state(xb), obs_vector=obs, obs_error_sd=np.full((nt + 1, no), 0.05), start_time=0.0,
            n_cycles=N_CYCLES, analysis_window=float(S - 1), model_inputs=jnp.asarray(f))
