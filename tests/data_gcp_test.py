"""Tests for GCP class (dabench.data.gcp)

Does NOT include model-level-moisture because that runs
    quite slow"""

from dabench.data import GCP
import pytest
import numpy as np


@pytest.fixture(scope='module')
def gcp_small():
    """Defines gcp object for rest of tests"""
    gcp_obj = GCP(date_start='2010-01-01T00:00:00', date_end='2010-01-01T05:00:00')
    ds = gcp_obj.load()

    return ds


@pytest.fixture(scope='module')
def gcp_multivar():
    """Defines gcp object for rest of tests"""
    gcp_obj = GCP(
            variables=['mean_sea_level_pressure', 'sea_surface_temperature'],
            date_start='1999-12-31T23:00:00', date_end='2000-01-01T00:00:00')
    ds = gcp_obj.load()

    return ds


def test_shapes(gcp_small):
    """Tests initialization of gcp data"""

    assert gcp_small.sizes['time'] == 6
    assert gcp_small.system_dim == 559
    assert gcp_small.dab.flatten().shape == (6, 559)
    assert gcp_small.to_array().shape == (1, 6, 13, 43)


def test_times(gcp_small):
    """Tests times"""
    assert gcp_small['time'].shape == (6,)
    assert gcp_small['time'].data[5] == np.datetime64('2010-01-01T05:00:00.000000000')


def test_values(gcp_small):
    """Tests values"""
    assert gcp_small.dab.flatten().values[0, 0] == pytest.approx(298.15393)
    assert gcp_small.dab.flatten().values[-1, 7] == pytest.approx(293.88577)


def test_shapes_multivar(gcp_multivar):
    """Tests shapes for multivariable object"""
    assert gcp_multivar.dab.flatten().shape == (2, 1118)
    assert gcp_multivar.sizes['time'] == 2 
    assert gcp_multivar.system_dim == 1118
    assert gcp_multivar.to_array().shape == (2, 2, 48, 13, 43)


def test_values_multivar(gcp_multivar):
    """Tests values for multivariable object"""
    assert gcp_multivar.dab.flatten().values[1, 25] == pytest.approx(101904.76563)
    assert gcp_multivar.dab.flatten().values[0, -1] == pytest.approx(298.68307)
