"""Tests for DataAzure class (dabench.data.azure)"""

from dabench.data.azure import DataAzure
import jax.numpy as jnp
import pytest
import numpy as np


@pytest.fixture
def azure_small():
    """Defines azure object for rest of tests"""
    azure_obj = DataAzure(date_start='1987-01-01', date_end='1987-01-05')
    azure_obj.load()

    return azure_obj


@pytest.fixture
def azure_multivar():
    """Defines azure object for rest of tests"""
    azure_obj2 = DataAzure(variables=['air_temperature_at_2_metres',
                                      'air_pressure_at_mean_sea_level'],
                           date_start='1993-06-10', date_end='1993-06-12')
    azure_obj2.load()

    return azure_obj2


def test_shapes(azure_small):
    """Tests initialization of azure data"""

    assert azure_small.original_dim == (13, 43)
    assert azure_small.time_dim == 120
    assert azure_small.system_dim == 559
    assert azure_small.values.shape == (120, 559)


def test_times(azure_small):
    """Tests times"""
    assert azure_small.times.shape == (120,)
    assert azure_small.times[9] == np.datetime64('1987-01-01T09:00:00.000000000')


def test_values(azure_small):
    """Tests values"""
    assert azure_small.values[31, 35] == 297.875
    assert azure_small.values[78, 237] == 293.6875


def test_shapes_multivar(azure_multivar):
    """Tests shapes for multivariable object"""
    assert azure_multivar.values.shape == (72, 1118)
    assert azure_multivar.original_dim == (13, 43, 2)
    assert azure_multivar.time_dim == 72
    assert azure_multivar.system_dim == 1118


def test_values_multivar(azure_multivar):
    """Tests values for multivariable object"""
    assert azure_multivar.values[-7, 500] == pytest.approx(101651.625)


def test_to_og_dim(azure_small, azure_multivar):
    """Tests to make sure reshape to original dim works"""
    assert azure_multivar.to_original_dim().shape == (72, 13, 43, 2)
    assert azure_multivar.to_original_dim()[9, 5, 5, 0] == pytest.approx(101352.0)
    assert azure_small.to_original_dim().shape == (120, 13, 43)
    assert azure_small.to_original_dim()[88, 5, 5] == 296.875

