"""Tests for DataAWS class (dabench.data.aws)"""

from dabench.data.aws import DataAWS
import jax.numpy as jnp
import pytest
import numpy as np


@pytest.fixture
def aws_small():
    """Defines aws object for rest of tests"""
    aws_obj = DataAWS(months=['01', '02'])
    aws_obj.load()

    return aws_obj

@pytest.fixture
def aws_multivar():
    """Defines aws object for rest of tests"""
    aws_obj = DataAWS(variables=['air_temperature_at_2_metres',
                                 'air_pressure_at_mean_sea_level'],
                      months=['01'])
    aws_obj.load()

    return aws_obj


def test_shapes(aws_small):
    """Tests initialization of aws data"""

    assert aws_small.original_dim == (13, 43)
    assert aws_small.time_dim == 1440
    assert aws_small.system_dim == 559
    assert aws_small.values.shape == (1440, 559)


def test_times(aws_small):
    """Tests times"""
    assert aws_small.times.shape == (1440,)
    assert aws_small.times[9] == np.datetime64('2020-01-01T09:00:00.000000000')


def test_values(aws_small):
    """Tests values"""
    assert aws_small.values[1300, 35] == 296.25
    assert aws_small.values[1200, 237] == 300.4375


def test_shapes_multivar(aws_multivar):
    """Tests shapes for multivariable object"""
    assert aws_multivar.values.shape == (744, 1118)
    assert aws_multivar.original_dim == (13, 43, 2)
    assert aws_multivar.time_dim == 744
    assert aws_multivar.system_dim == 1118


def test_values_multivar(aws_multivar):
    """Tests values for multivariable object"""
    assert aws_multivar.values[500, 500] == 101283.25


def test_to_og_dim(aws_small, aws_multivar):
    """Tests to make sure reshape to original dim works"""
    assert aws_multivar.to_original_dim().shape == (744, 13, 43, 2)
    assert aws_multivar.to_original_dim()[500, 5, 5, 1] == 295.375
    assert aws_small.to_original_dim().shape == (1440, 13, 43)
    assert aws_small.to_original_dim()[500, 5, 5] == 295.375

