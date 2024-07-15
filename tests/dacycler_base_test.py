"""Tests for base Data Assimilation Cycler class (dabench.dacycler._dacycler)"""

import pytest
import dabench as dab


def test_dacycler_init():
    """Tests initialization of dacycler"""

    params = {'system_dim': 6,
              'delta_t': 0.5,
              'ensemble': True}

    test_dac = dab.dacycler.DACycler(**params)

    assert test_dac.system_dim == 6
    assert test_dac.delta_t == 0.5
    assert test_dac.ensemble
    assert not test_dac.in_4d
