"""Tests for base Data class (dabench.data._data)"""

import datetime

import pytest
import numpy as np
import jax.numpy as jnp
import jaxlib

from dabench.data import Data


def test_data_init():
    """Test the initialization of class_data"""

    params = {'system_dim': 2,
              'time_dim': 8}

    test_data = Data(**params)

    assert test_data.system_dim == 2
    assert test_data.time_dim == 8


def test_set_values():
    """Test manually setting data values"""

    test_data = Data()

    x_test = np.arange(15).reshape(3, 5)
    test_data.values = x_test

    assert np.array_equal(test_data.values, x_test)


def test_set_values_jax():
    """Tests storing values as jax array"""

    test_data = Data(store_as_jax=True)

    x_test = np.arange(15).reshape(3, 5)
    test_data.values = x_test

    assert isinstance(test_data.values, jaxlib.xla_extension.DeviceArray)
    assert jnp.array_equal(test_data.values, x_test)


def test_to_original_dims():
    """Test returning data to original dimensions"""

    test_data = Data(original_dim=(2, 3))

    x_test = np.arange(18).reshape(3, 6)
    x_original = np.arange(18).reshape(3, 2, 3)
    test_data.values = x_test
    test_data.time_dim = x_test.shape[0]
    test_data.system_dim = x_test.shape[1]

    values_original_dim = test_data._to_original_dim()

    assert np.array_equal(x_original, values_original_dim)
    assert np.array_equal(
        test_data.values,
        values_original_dim.reshape(
            test_data.time_dim,
            test_data.system_dim)
        )


def test_load_netcdf():
    """Tests loading default netcdf (ERA5 ECWMF SLP)"""
    test_data = Data()
    test_data.load_netcdf()
    og_dim_data = test_data._to_original_dim()

    assert test_data.values.shape == (48, 3835)
    assert og_dim_data.shape == (48, 59, 65)
    assert og_dim_data[5, 5, 5] == pytest.approx(100588.86)


def test_load_netcdf_years():
    """Tests loading netcdf with only subset of years"""
    test_data = Data()
    test_data.load_netcdf(years_select=[2018, 2020])
    og_dim_data = test_data._to_original_dim()

    assert test_data.values.shape == (24, 3835)
    assert og_dim_data.shape == (24, 59, 65)
    assert og_dim_data[20, 5, 5] == pytest.approx(101301.56)
    with pytest.raises(
            ValueError,
            match='Dataset does not contain any of the years specified'
            ):
        test_data.load_netcdf(years_select=[1957, 2057])


def test_load_netcdf_dates():
    """Tests loading netcdf with only subset of dates"""
    test_data = Data()
    test_data.load_netcdf(dates_select=[datetime.date(2018, 1, 1),
                                        datetime.date(2018, 11, 1),
                                        datetime.date(2021, 5, 1)])
    og_dim_data = test_data._to_original_dim()

    assert test_data.values.shape == (3, 3835)
    assert og_dim_data.shape == (3, 59, 65)
    assert og_dim_data[2, 3, 4] == pytest.approx(100478.59)
    with pytest.raises(
            ValueError,
            match='Dataset does not contain any of the dates specified'
            ):
        test_data.load_netcdf(dates_select=[datetime.date(2018, 1, 31),
                                            datetime.date(1957, 11, 1)])
