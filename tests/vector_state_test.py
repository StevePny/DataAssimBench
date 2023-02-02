"""Tests for StateVector class (dabench.vector._state_vector)"""

import pytest
import numpy as np
import jax.numpy as jnp
import jaxlib

from dabench.vector import StateVector


@pytest.fixture
def single_sv():
    """Defines single timestep StateVector"""
    params = {'system_dim': 2,
              'time_dim': 1,
              'values': np.array([[1, 2]])
              }

    return StateVector(**params)


@pytest.fixture
def single_sv_jax():
    """Defines single timestep StateVector using Jax backend"""
    params = {'system_dim': 2,
              'time_dim': 1,
              'values': np.array([[1, 2]]),
              'store_as_jax': True
              }

    return StateVector(**params)


@pytest.fixture
def sv_trajectory():
    """Defines 8-timestep state vector trajectory"""
    params = {'system_dim': 2,
              'time_dim': 8,
              'values':  np.arange(0,  16).reshape((8, 2))}

    return StateVector(**params)


def test_init(single_sv):
    """Test the initialization of StateVector"""

    assert single_sv.system_dim == 2
    assert single_sv.time_dim == 1
    assert np.array_equal(single_sv.values, np.array([[1, 2]]))
    assert np.array_equal(single_sv.xi, single_sv.values[0])


def test_init_jax(single_sv_jax):
    """Test the initialization of StateVector with Jax backend"""

    assert single_sv_jax.system_dim == 2
    assert single_sv_jax.time_dim == 1
    assert jnp.array_equal(single_sv_jax.values, jnp.array([[1, 2]]))
    assert jnp.array_equal(single_sv_jax.xi, single_sv_jax.values[0])


def test_init_traj(sv_trajectory):
    """Test initialization of StateVector trajectory (time_dim==8)"""

    assert sv_trajectory.system_dim == 2
    assert sv_trajectory.time_dim == 8
    assert np.array_equal(sv_trajectory.values[3], np.array([6, 7]))
    assert np.array_equal(sv_trajectory.xi, sv_trajectory.values[-1])


def test_set_values():
    """Test manually setting data values"""

    test_vec = StateVector()

    new_values = np.arange(15).reshape(3, 5)
    test_vec.values = new_values

    assert np.array_equal(test_vec.values, new_values)
    assert np.array_equal(test_vec.xi, new_values[-1])

# 
# 
# def test_set_values_jax():
#     """Tests storing values as jax array"""
# 
#     test_data = Data(store_as_jax=True)
# 
#     x_test = np.arange(15).reshape(3, 5)
#     test_data.values = x_test
# 
#     assert isinstance(test_data.values, jaxlib.xla_extension.DeviceArray)
#     assert jnp.array_equal(test_data.values, x_test)
# 
# 
# def test_to_original_dims():
#     """Test returning data to original dimensions"""
# 
#     test_data = Data(original_dim=(2, 3))
# 
#     x_test = np.arange(18).reshape(3, 6)
#     x_original = np.arange(18).reshape(3, 2, 3)
#     test_data.values = x_test
#     test_data.time_dim = x_test.shape[0]
#     test_data.system_dim = x_test.shape[1]
# 
#     values_original_dim = test_data._to_original_dim()
# 
#     assert np.array_equal(x_original, values_original_dim)
#     assert np.array_equal(
#         test_data.values,
#         values_original_dim.reshape(
#             test_data.time_dim,
#             test_data.system_dim)
#         )
# 
# 
# def test_load_netcdf():
#     """Tests loading default netcdf (ERA5 ECWMF SLP)"""
#     test_data = Data()
#     test_data.load_netcdf()
#     og_dim_data = test_data._to_original_dim()
# 
#     assert test_data.values.shape == (48, 3835)
#     assert og_dim_data.shape == (48, 59, 65)
#     assert og_dim_data[5, 5, 5] == pytest.approx(100588.86)
# 
# 
# def test_load_netcdf_years():
#     """Tests loading netcdf with only subset of years"""
#     test_data = Data()
#     test_data.load_netcdf(years_select=[2018, 2020])
#     og_dim_data = test_data._to_original_dim()
# 
#     assert test_data.values.shape == (24, 3835)
#     assert og_dim_data.shape == (24, 59, 65)
#     assert og_dim_data[20, 5, 5] == pytest.approx(101301.56)
#     with pytest.raises(
#             ValueError,
#             match='Dataset does not contain any of the years specified'
#             ):
#         test_data.load_netcdf(years_select=[1957, 2057])
# 
# 
# def test_load_netcdf_dates():
#     """Tests loading netcdf with only subset of dates"""
#     test_data = Data()
#     test_data.load_netcdf(dates_select=[datetime.date(2018, 1, 1),
#                                         datetime.date(2018, 11, 1),
#                                         datetime.date(2021, 5, 1)])
#     og_dim_data = test_data._to_original_dim()
# 
#     assert test_data.values.shape == (3, 3835)
#     assert og_dim_data.shape == (3, 59, 65)
#     assert og_dim_data[2, 3, 4] == pytest.approx(100478.59)
#     with pytest.raises(
#             ValueError,
#             match='Dataset does not contain any of the dates specified'
#             ):
#         test_data.load_netcdf(dates_select=[datetime.date(2018, 1, 31),
#                                             datetime.date(1957, 11, 1)])
