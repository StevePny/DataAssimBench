"""The 4D-Var health line (`health=True`): one line per outer loop and cycle with the cost terms, from inside the compiled window; the analysis is
the same as without it."""
import os
import sys

import jax
import jax.numpy as jnp
import numpy as np

import dabench as dab

sys.path.insert(0, os.path.dirname(__file__))
from dacycler_compiled_window_test import LOCS, N_CYCLES, S, _inputs_cycler, _setup, obs_vector, state  # noqa: E402

jax.config.update("jax_enable_x64", True)


def test_health_lines_and_unchanged_analysis(capfd):
    f, truth, xb = _setup()
    common = dict(input_state=state(xb), start_time=0.0, obs_vector=obs_vector(truth, LOCS), obs_error_sd=0.05, n_cycles=N_CYCLES,
                  analysis_window=float(S - 1), return_forecast=True, model_inputs=jnp.asarray(f))
    kw = dict(Q_half=True, eta_mode="constant")
    a0 = np.asarray(_inputs_cycler(dab.dacycler.Var4DOperatorWC, **kw).cycle(**common)['x'].data)
    capfd.readouterr()
    a1 = np.asarray(_inputs_cycler(dab.dacycler.Var4DOperatorWC, health=True, **kw).cycle(**common)['x'].data)
    out = capfd.readouterr().out
    lines = [ln for ln in out.splitlines() if ln.startswith("[4dvar-health]")]
    assert len(lines) == 2 * N_CYCLES and "outer 0" in lines[0] and "outer 1" in lines[1] and "Jo" in lines[0]
    np.testing.assert_allclose(a1, a0, rtol=1e-12, atol=1e-14)


def test_jo_by_type_sums_to_jo(capfd):
    import re
    from dacycler_compiled_window_test import N
    f, truth, xb = _setup()
    common = dict(input_state=state(xb), start_time=0.0, obs_vector=obs_vector(truth, LOCS), obs_error_sd=0.05, n_cycles=N_CYCLES,
                  analysis_window=float(S - 1), return_forecast=True, model_inputs=jnp.asarray(f))
    groups = [("first", 0, N // 2), ("second", N // 2, N)]
    _inputs_cycler(dab.dacycler.Var4DOperatorWC, health=True, obs_groups=groups, Q_half=True, eta_mode="constant").cycle(**common)
    out = [ln for ln in capfd.readouterr().out.splitlines() if ln.startswith("[4dvar-health]")]
    main = [ln for ln in out if "Jo by type" not in ln]
    by = [ln for ln in out if "Jo by type" in ln]
    assert len(by) == len(main) == 2 * N_CYCLES
    for m, b in zip(main, by):
        jo = float(re.search(r"Jo ([0-9.e+-]+) \(2Jo", m).group(1))
        p = int(re.search(r"p (\d+)\)", m).group(1))
        parts = [float(x) for x in re.findall(r"(?:first|second) ([0-9.e+-]+)", b)]
        ps = [int(x) for x in re.findall(r"\(p (\d+)\)", b)]
        np.testing.assert_allclose(sum(parts), jo, rtol=1e-3)
        assert sum(ps) == p
