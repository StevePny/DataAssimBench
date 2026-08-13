"""Cross-cycler obs-space metrics regression test.

Runs every FGAT / 4D / Var cycler on ONE shared nature run, observation set,
window, and ensemble, then pins their per-cycle obs-space diagnostics
(``o_minus_f_rms``, ``o_minus_a_rms``, the FGAT tau-restricted pair, the
end-of-window O-A, ``bias_a``, ``obs_space_spread_background``) to reference
values so a
change in relative cycler performance is caught.  Same obs count / obs error /
window for all methods, so the numbers are directly comparable.
"""

import pytest
import numpy as np
import jax
jax.config.update("jax_enable_x64", True)
import jax.random as jrand
import dabench as dab

KEY = jrand.PRNGKey(42)
ENS = 8
SYS = 5
OBS_SD = 1.0
N_CYCLES = 10
RTOL = 2e-3
ATOL = 5e-3

KEYS = ("o_minus_f_rms", "o_minus_a_rms", "o_minus_f_rms_at_tau",
        "o_minus_a_rms_at_tau", "o_minus_a_rms_end", "bias_a",
        "obs_space_spread_background", "obs_space_spread_analysis_end")

# Pinned references (fp64, KEY=42, seed=91).  Regenerate deliberately if the
# analysis convention changes; a spurious drift here is a real regression.
REF = {
    "ETKF-FGAT": {"o_minus_f_rms": 0.997618, "o_minus_a_rms": 1.251992, "o_minus_f_rms_at_tau": 1.012419, "o_minus_a_rms_at_tau": 0.933418, "o_minus_a_rms_end": 0.874192, "bias_a": 0.058085, "obs_space_spread_background": 0.607984, "obs_space_spread_analysis_end": 0.500716},
    "LETKF-FGAT": {"o_minus_f_rms": 0.987392, "o_minus_a_rms": 1.244624, "o_minus_f_rms_at_tau": 0.976019, "o_minus_a_rms_at_tau": 0.893873, "o_minus_a_rms_end": 0.882789, "bias_a": 0.071981, "obs_space_spread_background": 0.663737, "obs_space_spread_analysis_end": 0.553830},
    "ETKF4D": {"o_minus_f_rms": 0.986989, "o_minus_a_rms": 1.578653, "o_minus_f_rms_at_tau": np.nan, "o_minus_a_rms_at_tau": np.nan, "o_minus_a_rms_end": 0.824635, "bias_a": 0.073255, "obs_space_spread_background": 0.441191, "obs_space_spread_analysis_end": 0.322882},
    "LETKF4D": {"o_minus_f_rms": 0.994059, "o_minus_a_rms": 1.565103, "o_minus_f_rms_at_tau": np.nan, "o_minus_a_rms_at_tau": np.nan, "o_minus_a_rms_end": 0.823805, "bias_a": 0.093729, "obs_space_spread_background": 0.488040, "obs_space_spread_analysis_end": 0.361861},
    "Var3D-FGAT": {"o_minus_f_rms": 0.921884, "o_minus_a_rms": 1.199126, "o_minus_f_rms_at_tau": 0.868889, "o_minus_a_rms_at_tau": 0.774914, "o_minus_a_rms_end": 0.823147, "bias_a": -0.017134, "obs_space_spread_background": 1.000000, "obs_space_spread_analysis_end": 0.707107},
    "Var4D": {"o_minus_f_rms": 0.939480, "o_minus_a_rms": 0.719685, "o_minus_f_rms_at_tau": np.nan, "o_minus_a_rms_at_tau": np.nan, "o_minus_a_rms_end": 0.722639, "bias_a": -0.021682, "obs_space_spread_background": 1.000000, "obs_space_spread_analysis_end": 0.553588},
}

# Deterministic (variational) cyclers: spread is B-derived (static-B analogue
# of the ensemble spread) rather than ensemble-estimated.
DETERMINISTIC = {"Var3D-FGAT", "Var4D"}
# Cyclers emitting the FGAT tau-restricted comparison pair.
FGAT = {"ETKF-FGAT", "LETKF-FGAT", "Var3D-FGAT"}


@pytest.fixture(scope="module")
def nature():
    l96 = dab.data.Lorenz96(system_dim=SYS, store_as_jax=True, delta_t=0.01)
    return l96.generate(n_steps=120)


@pytest.fixture(scope="module")
def obs_vec(nature):
    return dab.observer.Observer(
        nature, times=nature["time"].data[np.arange(0, 120, 5)],
        random_location_count=3, error_bias=0.0, error_sd=OBS_SD,
        random_seed=91, stationary_observers=True, store_as_jax=True).observe()


@pytest.fixture(scope="module")
def fc_model():
    m = dab.data.Lorenz96(system_dim=SYS, store_as_jax=True, delta_t=0.01)

    class L96Model(dab.model.Model):
        def forecast(self, state_vec, n_steps):
            v = self.model_obj.generate(x0=state_vec["x"].data, n_steps=n_steps)
            return v.isel(time=-1).assign_attrs(delta_t=0.01), v

        def compute_tlm(self, state_vec, n_steps):
            x, M = self.model_obj.generate(
                n_steps=n_steps, x0=state_vec["x"].data, return_tlm=True)
            return x, M

    return L96Model(model_obj=m)


def _init_ens(nature):
    s = nature.isel(time=10)
    return s.assign(x=(["ensemble", "index"],
                       s["x"].data + jrand.normal(KEY, shape=(ENS, SYS))))


def _init_det(nature):
    s = nature.isel(time=10)
    return s.assign(x=(["index"], s["x"].data + 0.3))


def _cycler(name, fc_model):
    C = dab.dacycler
    if name == "ETKF-FGAT":
        return C.ETKF(system_dim=SYS, delta_t=0.01, ensemble_dim=ENS,
                      model_obj=fc_model, fgat=True, analysis_time_index="mid")
    if name == "LETKF-FGAT":
        return C.LETKF(system_dim=SYS, delta_t=0.01, ensemble_dim=ENS,
                       model_obj=fc_model, localize_radius=2.0, fgat=True,
                       analysis_time_index="mid")
    if name == "ETKF4D":
        return C.ETKF4D(system_dim=SYS, delta_t=0.01, ensemble_dim=ENS,
                        model_obj=fc_model)
    if name == "LETKF4D":
        return C.LETKF4D(system_dim=SYS, delta_t=0.01, ensemble_dim=ENS,
                         model_obj=fc_model, localize_radius=2.0)
    if name == "Var3D-FGAT":
        return C.Var3D(system_dim=SYS, delta_t=0.01, model_obj=fc_model,
                       fgat=True, analysis_time_index="mid")
    if name == "Var4D":
        return C.Var4D(system_dim=SYS, delta_t=0.01, model_obj=fc_model,
                       obs_window_indices=[0, 5, 10], steps_per_window=11)
    raise KeyError(name)


def _agg(metrics, key):
    if key not in metrics:
        return float("nan")
    v = np.asarray(metrics[key].data)
    v = v[np.isfinite(v)]
    return float(np.mean(v)) if v.size else float("nan")


def _run(name, nature, obs_vec, fc_model):
    cyc = _cycler(name, fc_model)
    init = _init_det(nature) if name in DETERMINISTIC else _init_ens(nature)
    ana, metrics = cyc.cycle(
        input_state=init, start_time=init["time"].data, obs_vector=obs_vec,
        obs_error_sd=OBS_SD, analysis_window=0.1, n_cycles=N_CYCLES,
        return_forecast=True, return_metrics=True)
    return cyc, metrics


@pytest.mark.parametrize("name", list(REF))
def test_cross_cycler_metrics_pinned(name, nature, obs_vec, fc_model):
    """Pinned per-cycle obs-space metrics + structural invariants per cycler."""
    cyc, metrics = _run(name, nature, obs_vec, fc_model)

    # Instance-stored + reshapeable + memory-tracked container.
    assert cyc.metrics is not None
    assert int(cyc.metrics.sizes["cycle"]) == N_CYCLES
    assert cyc.metrics.nbytes > 0

    for k in KEYS:
        got = _agg(metrics, k)
        exp = REF[name][k]
        if np.isnan(exp):
            assert np.isnan(got), f"{name}/{k}: expected NaN, got {got}"
        else:
            assert got == pytest.approx(exp, rel=RTOL, abs=ATOL), \
                f"{name}/{k}: {got} != {exp}"

    # Spread contraction invariant (both keys are pinned above, so their values
    # are already asserted there): the analysis-end spread (next-cycle IC
    # spread) is SMALLER than the background spread -- the analysis contracts
    # the covariance.  Holds for every cycler now that the deterministic
    # (variational) methods emit a B-derived posterior spread.
    spread_bg = _agg(metrics, "obs_space_spread_background")
    spread_ana_end = _agg(metrics, "obs_space_spread_analysis_end")
    assert spread_ana_end < spread_bg

    # End-of-window O-A (next-cycle IC quality) is always finite here.
    assert np.isfinite(_agg(metrics, "o_minus_a_rms_end"))

    # FGAT tau-restricted pair: O-A <= O-F at the analysis time (the
    # like-for-like fit diagnostic).  4D/Var4D do not emit the pair (NaN).
    if name in FGAT:
        of_tau = _agg(metrics, "o_minus_f_rms_at_tau")
        oa_tau = _agg(metrics, "o_minus_a_rms_at_tau")
        assert oa_tau <= of_tau + 1e-8
    else:
        assert np.isnan(_agg(metrics, "o_minus_f_rms_at_tau"))
