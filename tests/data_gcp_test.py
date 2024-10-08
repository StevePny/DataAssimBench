"""Tests for GCP class (dabench.data.gcp)

Does NOT include model-level-moisture because that runs
    quite slow"""

from dabench.data import GCP
import pytest
import numpy as np


@pytest.fixture(scope='module')
def gcp_small():
    """Defines gcp object for rest of tests"""
    gcp_obj = GCP(date_start='2010-01-01', date_end='2010-01-03')
    gcp_obj.load()

    return gcp_obj


@pytest.fixture(scope='module')
def gcp_multivar():
    """Defines gcp object for rest of tests"""
    gcp_obj = GCP(
            variables=['mean_sea_level_pressure', 'sea_surface_temperature'],
            date_start='1999-12-31', date_end='2000-01-01')
    gcp_obj.load()

    return gcp_obj


def test_shapes(gcp_small):
    """Tests initialization of gcp data"""

    assert gcp_small.time_dim == 72
    assert gcp_small.system_dim == 559
    assert gcp_small.values.shape == (72, 559)
    assert gcp_small.original_dim == (13, 43)
    assert gcp_small.values_gridded.shape == (72, 13, 43)


def test_times(gcp_small):
    """Tests times"""
    assert gcp_small.times.shape == (72,)
    assert gcp_small.times[5] == np.datetime64('2010-01-01T05:00:00.000000000')


def test_values(gcp_small):
    """Tests values"""
    assert gcp_small.values[0, 0] == pytest.approx(298.15393)
    assert gcp_small.values[-22, 7] == pytest.approx(289.72427)


def test_shapes_multivar(gcp_multivar):
    """Tests shapes for multivariable object"""
    assert gcp_multivar.values.shape == (48, 1118)
    assert gcp_multivar.time_dim == 48 
    assert gcp_multivar.system_dim == 1118
    assert gcp_multivar.original_dim == (13, 43, 2)
    assert gcp_multivar.values_gridded.shape == (48, 13, 43, 2)


def test_values_multivar(gcp_multivar):
    """Tests values for multivariable object"""
    assert gcp_multivar.values[1, 1000] == pytest.approx(101753.71)
    assert gcp_multivar.values[0, -1] == pytest.approx(298.61407)
