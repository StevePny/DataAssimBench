"""Tests for base Data Assimilation Cycler class (dabench.dacycler._dacycler)"""

import pytest
import numpy as np
import dabench as dab
from dabench.dacycler import _utils as dac_utils


def test_dacycler_init():
    """Tests initialization of dacycler"""

    params = {'system_dim': 6,
              'delta_t': 0.5,
              'model_obj':dab.model.RCModel(6, 10)}

    test_dac = dab.dacycler.DACycler(**params)

    assert test_dac.system_dim == 6
    assert test_dac.delta_t == 0.5
    assert not test_dac._uses_ensemble
    assert not test_dac._in_4d


@pytest.mark.parametrize("analysis_cycles", list(range(1, 25)))
@pytest.mark.parametrize("analysis_window", [0.1, 0.05, 0.25, 0.5, 1.0, 6.0])
@pytest.mark.parametrize("start_time", [0.0, 3.0, 1.5])
def test_get_all_times_length_invariant(
        start_time, analysis_window, analysis_cycles):
    """``_get_all_times`` returns EXACTLY ``analysis_cycles`` window times.

    Regression guard for the ``jnp.arange(0, N*window, window)`` float-endpoint
    bug: for ``window=0.1`` at ``N in {3, 6, 12}`` (and other combos) floating-
    point rounding pushed ``N*window`` just above the last multiple, so
    ``arange`` returned ``N+1`` elements.  That over-long schedule then either
    mismatched the length-N ``repeat`` addend (broadcast error) or silently
    shifted every downstream obs window.  The integer-index form
    ``arange(N)*window`` must be exactly N for any window.
    """
    times = dac_utils._get_all_times(
        start_time, analysis_window, analysis_cycles)
    assert times.shape == (analysis_cycles,), (
        f"expected {analysis_cycles} window times, got {times.shape[0]} "
        f"(start={start_time}, window={analysis_window})")
    # Window centers are the arithmetic progression start + k*window.
    expected = start_time + np.arange(analysis_cycles) * analysis_window
    assert np.allclose(np.asarray(times), expected, rtol=0, atol=0)
