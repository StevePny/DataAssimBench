"""Tests for ObsOp class (dabench.obsop._obsop)"""

import pytest
import numpy as np
import jax.numpy as jnp

from dabench import vector, obsop


@pytest.fixture
def single_sv():
    """Defines single timestep StateVector"""
    params = {'system_dim': 2,
              'time_dim': 1,
              'values': np.array([[1, 2]])
              }

    return vector.StateVector(**params)


@pytest.fixture
def single_sv_jax():
    """Defines single timestep StateVector using Jax backend"""
    params = {'system_dim': 2,
              'time_dim': 1,
              'values': np.array([[1, 2]]),
              'store_as_jax': True
              }

    return vector.StateVector(**params)


@pytest.fixture
def sv_trajectory():
    """Defines 8-timestep state vector trajectory"""
    params = {'system_dim': 3,
              'time_dim': 8,
              'values':  np.arange(0,  24).reshape((8, 3))}

    return vector.StateVector(**params)


@pytest.fixture
def obsop_default():
    """Defines simple ObsOp using default settings."""
    return obsop.ObsOp()


@pytest.fixture
def obsop_manual_locindices():
    """Defines ObsOp with manually specified loc indices"""
    return obsop.ObsOp(location_indices=[0])


@pytest.fixture
def obsop_random_locindices_1():
    """Defines ObsOp with 1 randomly generated loc indices"""
    return obsop.ObsOp(random_location_count=1)


@pytest.fixture
def obsop_random_locindices_2():
    """Defines ObsOp with 2 randomly generated loc indices"""
    return obsop.ObsOp(random_location_count=2)


@pytest.fixture
def obsop_custom_H():
    """Defines ObsOp with custom H operator"""
    return obsop.ObsOp(
            H=np.array([[5., 2.], [0., 1.]])
            )


def test_defaults_singlesv(single_sv, obsop_default):
    """Test default settings on single timestep"""

    obs_vec = obsop_default.observe(single_sv)

    assert obs_vec.obs_dims == np.array([2])
    assert obs_vec.time_dim == 1
    assert np.array_equal(obs_vec.times, np.array([0]))
    assert np.array_equal(obs_vec.location_indices, np.array([0, 1]))
    assert np.array_equal(obs_vec.values, np.array([[1., 2.]]))
    assert isinstance(obs_vec.values, np.ndarray)


def test_defaults_singlesv_jax(single_sv_jax, obsop_default):
    obs_vec = obsop_default.observe(single_sv_jax)

    assert obs_vec.obs_dims == np.array([2])
    assert obs_vec.time_dim == 1
    assert np.array_equal(obs_vec.times, np.array([0]))
    assert np.array_equal(obs_vec.location_indices, np.array([0, 1]))
    assert np.array_equal(obs_vec.values,
                          jnp.array([[1., 2.]]))
    assert isinstance(obs_vec.values, jnp.ndarray)


def test_defaults_trajectory(sv_trajectory, obsop_default):
    """Test default settings on trajectory (multiple timesteps)"""

    obs_vec = obsop_default.observe(sv_trajectory)

    assert np.array_equal(obs_vec.obs_dims, np.repeat(3, 8))
    assert obs_vec.time_dim == 8
    assert np.array_equal(obs_vec.times, np.arange(8))
    assert np.array_equal(obs_vec.location_indices, np.array([0, 1, 2]))
    assert np.array_equal(obs_vec.values,  np.arange(0, 24).reshape((8, 3)))
    assert isinstance(obs_vec.values, np.ndarray)


def test_man_locinds_singlesv(single_sv, obsop_manual_locindices):
    """Test manually specified indices on single timestep"""

    obs_vec = obsop_manual_locindices.observe(single_sv)

    assert obs_vec.obs_dims == np.array([1])
    assert obs_vec.time_dim == 1
    assert np.array_equal(obs_vec.times, np.array([0]))
    assert np.array_equal(obs_vec.location_indices, np.array([0]))
    assert np.array_equal(obs_vec.values, np.array([[1.]]))
    assert isinstance(obs_vec.values, np.ndarray)


def test_man_locinds_trajectory(sv_trajectory, obsop_manual_locindices):
    """Test manually specified indices on trajectory"""

    obs_vec = obsop_manual_locindices.observe(sv_trajectory)

    assert np.array_equal(obs_vec.obs_dims, np.repeat(1, 8))
    assert obs_vec.time_dim == 8
    assert np.array_equal(obs_vec.times, np.arange(8))
    assert np.array_equal(obs_vec.location_indices, np.array([0]))
    assert np.array_equal(obs_vec.values,
                          np.arange(24, step=3, dtype='float').reshape(8, 1))
    assert isinstance(obs_vec.values, np.ndarray)


def test_random_locinds_singlesv(single_sv, obsop_random_locindices_1):
    """Test randomly generated indices on single timestep"""

    obs_vec = obsop_random_locindices_1.observe(single_sv)

    assert obs_vec.obs_dims == np.array([1])
    assert obs_vec.time_dim == 1
    assert np.array_equal(obs_vec.times, np.array([0]))
    assert np.array_equal(obs_vec.location_indices, np.array([1]))
    assert np.array_equal(obs_vec.values, np.array([[2.]]))
    assert isinstance(obs_vec.values, np.ndarray)


def test_random_locinds_trajectory(sv_trajectory, obsop_random_locindices_2):
    """Test randomly generated indices on trajectory"""

    obs_vec = obsop_random_locindices_2.observe(sv_trajectory)

    assert np.array_equal(obs_vec.obs_dims, np.repeat(2, 8))
    assert obs_vec.time_dim == 8
    assert np.array_equal(obs_vec.times, np.arange(8))
    assert np.array_equal(obs_vec.location_indices, np.array([1, 2]))
    assert np.array_equal(obs_vec.values,
                          np.array([[1., 2.],
                                    [4., 5.],
                                    [7., 8.],
                                    [10., 11.],
                                    [13., 14.],
                                    [16., 17.],
                                    [19., 20.],
                                    [22., 23.]])
                          )
    assert isinstance(obs_vec.values, np.ndarray)


def test_custom_H_singlesv(single_sv, obsop_custom_H):
    """Test default settings on single timestep"""

    obs_vec = obsop_custom_H.observe(single_sv)

    assert obs_vec.obs_dims == np.array([2]) 
    assert obs_vec.time_dim == 1
    assert np.array_equal(obs_vec.times, np.array([0]))
    assert np.array_equal(obs_vec.values, np.array([[9., 2.]]))
    assert isinstance(obs_vec.values, np.ndarray)
