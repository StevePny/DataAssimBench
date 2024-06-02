"""Tests for GCP class (dabench.data.gcp)

Does NOT include model-level-moisture because that runs
    quite slow"""

from dabench.data import GCP
import pytest
import numpy as np


@pytest.fixture(scope='module')
def gcp_small():
    """Defines gcp object for rest of tests"""
    gcp_obj = GCP(data_type='single-level-reanalysis',
                  date_start='2010-06-05', date_end='2010-06-07')
    gcp_obj.load()

    return gcp_obj


@pytest.fixture(scope='module')
def gcp_multivar():
    """Defines gcp object for rest of tests"""
    gcp_obj = GCP(data_type='single-level-forecast',
                  variables=['cp', 'ssrd'],
                  date_start='1999-12-31', date_end='2000-01-01')
    gcp_obj.load()

    return gcp_obj


def test_shapes(gcp_small):
    """Tests initialization of gcp data"""

    assert gcp_small.time_dim == 72
    assert gcp_small.system_dim == 441
    assert gcp_small.values.shape == (72, 441)


def test_times(gcp_small):
    """Tests times"""
    assert gcp_small.times.shape == (72,)
    assert gcp_small.times[5] == np.datetime64('2010-06-05T05:00:00.000000000')


def test_values(gcp_small):
    """Tests values"""
    assert gcp_small.values[0, 0] == pytest.approx(302.12622)
    assert gcp_small.values[-22, 7] == pytest.approx(303.03052)


def test_shapes_multivar(gcp_multivar):
    """Tests shapes for multivariable object"""
    assert gcp_multivar.values.shape == (4, 16758)
    assert gcp_multivar.time_dim == 4
    assert gcp_multivar.system_dim == 16758


def test_values_multivar(gcp_multivar):
    """Tests values for multivariable object"""
    assert gcp_multivar.values[1, 2025] == 1588608.0
    assert gcp_multivar.values[3, -1] == 28288.0
