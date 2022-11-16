"""Tests for data class (dabench.data.data)"""

from dabench.data import data
import jax.numpy as jnp
import pytest
import datetime


def test_data_init():
    """Test the initialization of class_data"""

    params = {'system_dim': 2,
              'time_dim': 8}

    test_data = data.Data(**params)

    assert test_data.system_dim == 2
    assert test_data.time_dim == 8


def test_set_values():
    """Test manually setting data values"""

    test_data = data.Data()

    x_test = jnp.arange(15).reshape(3, 5)
    test_data.set_values(x_test)

    assert jnp.array_equal(test_data.values, x_test)


def test_to_original_dims():
    """Test returning data to original dimensions"""

    test_data = data.Data(original_dim=(2, 3))

    x_test = jnp.arange(18).reshape(3, 6)
    x_original = jnp.arange(18).reshape(3, 2, 3)
    test_data.set_values(x_test)

    values_original_dim = test_data.to_original_dim()

    assert jnp.array_equal(x_original, values_original_dim)
    assert jnp.array_equal(
        test_data.values,
        values_original_dim.reshape(
            test_data.time_dim,
            test_data.system_dim)
        )


def test_load_netcdf():
    """Tests loading default netcdf (ERA5 ECWMF SLP)"""
    test_data = data.Data()
    test_data.load_netcdf()
    og_dim_data = test_data.to_original_dim()

    assert test_data.values.shape == (48, 3835)
    assert og_dim_data.shape == (48, 59, 65)
    assert og_dim_data[5, 5, 5] == pytest.approx(100588.86)


def test_load_netcdf_years():
    """Tests loading netcdf with only subset of years"""
    test_data = data.Data()
    test_data.load_netcdf(years_select=[2018, 2020])
    og_dim_data = test_data.to_original_dim()

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
    test_data = data.Data()
    test_data.load_netcdf(dates_select=[datetime.date(2018, 1, 1),
                                        datetime.date(2018, 11, 1),
                                        datetime.date(2021, 5, 1)])
    og_dim_data = test_data.to_original_dim()

    assert test_data.values.shape == (3, 3835)
    assert og_dim_data.shape == (3, 59, 65)
    assert og_dim_data[2, 3, 4] == pytest.approx(100478.59)
    with pytest.raises(
            ValueError,
            match='Dataset does not contain any of the dates specified'
            ):
        test_data.load_netcdf(dates_select=[datetime.date(2018, 1, 31),
                                            datetime.date(1957, 11, 1)])
