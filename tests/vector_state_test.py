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
    """Test manually setting vector values"""

    test_vec = StateVector()

    new_values = np.arange(15).reshape(3, 5)
    test_vec.values = new_values

    assert np.array_equal(test_vec.values, new_values)
    assert np.array_equal(test_vec.xi, new_values[-1])


def test_set_values_jax():
    """Test manually setting vector values with jax backend"""

    test_vec = StateVector(store_as_jax=True)

    new_values = np.arange(15).reshape(3, 5)
    test_vec.values = new_values

    assert jnp.array_equal(test_vec.values, new_values)
    assert jnp.array_equal(test_vec.xi, new_values[-1])
    assert isinstance(test_vec.values, jaxlib.xla_extension.DeviceArray)


def test_to_original_dims():
    """Test returning data to original dimensions"""

    test_vec = StateVector(original_dim=(2, 3))

    test_values = np.arange(18).reshape(3, 6)
    values_og_dim = np.arange(18).reshape(3, 2, 3)
    test_vec.values = test_values
    test_vec.time_dim = test_values.shape[0]
    test_vec.system_dim = test_values.shape[1]

    assert np.array_equal(values_og_dim, test_vec.values_gridded)
    assert np.array_equal(
        test_vec.values,
        values_og_dim.reshape(
            test_vec.time_dim,
            test_vec.system_dim)
        )
