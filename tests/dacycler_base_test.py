"""Tests for base Data Assimilation Cycler class (dabench.dacycler._dacycler)"""

import pytest
import dabench as dab


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
