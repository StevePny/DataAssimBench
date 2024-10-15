"""Tests for base Data class (dabench.data._data)"""

import datetime

import pytest
import numpy as np
import jax.numpy as jnp
import xarray as xr

from dabench.data import Data


def test_data_init():
    """Test the initialization of class_data"""

    params = {'system_dim': 2}

    test_data = Data(**params)

    assert test_data.system_dim == 2


def test_load_netcdf():
    """Tests loading default netcdf (ERA5 ECWMF SLP)"""
    test_data = Data()
    test_nr = test_data.load_netcdf()

    assert test_nr.dab.flatten().shape == (48, 3835)
    assert test_nr.to_array().shape == (1, 48, 59, 65)
    assert test_nr.to_array()[0, 5, 5, 5] == pytest.approx(100588.86)


def test_load_netcdf_years():
    """Tests loading netcdf with only subset of years"""
    test_data = Data()
    test_nr = test_data.load_netcdf(years_select=[2018, 2020])

    assert test_nr.dab.flatten().shape == (24, 3835)
    assert test_nr.to_array().shape == (1, 24, 59, 65)
    assert test_nr.to_array()[0, 20, 5, 5] == pytest.approx(101301.56)
    with pytest.raises(
            ValueError,
            match='Dataset does not contain any of the years specified'
            ):
        test_data.load_netcdf(years_select=[1957, 2057])


def test_load_netcdf_dates():
    """Tests loading netcdf with only subset of dates"""
    test_data = Data()
    test_nr = test_data.load_netcdf(
            dates_select=[datetime.date(2018, 1, 1),
                          datetime.date(2018, 11, 1),
                          datetime.date(2021, 5, 1)])
    og_dim_data = test_nr.to_array()

    assert test_nr.dab.flatten().shape == (3, 3835)
    assert og_dim_data.shape == (1, 3, 59, 65)
    assert og_dim_data[0, 2, 3, 4] == pytest.approx(100478.59)
    with pytest.raises(
            ValueError,
            match='Dataset does not contain any of the dates specified'
            ):
        test_data.load_netcdf(dates_select=[datetime.date(2018, 1, 31),
                                            datetime.date(1957, 11, 1)])


def test_split_train_valid_test():
    """Tests splitting data object into train, validation, test sets"""

    x_test = np.arange(50).reshape(10, 5)
    test_nr = xr.Dataset(
            {'x': (('time', 'index'), x_test)}
            )

    s_train, s_val, s_test = test_nr.dab.split_train_val_test([5, 3, 2])

    assert s_train.sizes['time'] == 5
    assert s_val.sizes['time'] == 3
    assert s_test.sizes['time'] == 2


def test_split_train_valid_test_fracs():
    """Tests splitting data object using fractions for split sizes"""
    x_test = np.arange(50).reshape(10, 5)
    test_nr = xr.Dataset(
            {'x': (('time', 'index'), x_test)}
            )

    s_train, s_val, s_test = test_nr.dab.split_train_val_test([0.5, 0.3, 0.2])
    assert s_train.sizes['time'] == 5
    assert s_val.sizes['time'] == 3
    assert s_test.sizes['time'] == 2
