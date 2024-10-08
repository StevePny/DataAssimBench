"""Tests for ObsOp class (dabench.obsop._obsop)"""

import pytest
import numpy as np
import jax.numpy as jnp

from dabench import vector, obsop

pytest.skip('ObsOp not currently implemented', allow_module_level=True)

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


def custom_h(state_vec, obs_vec):
    """Custom h that sums the values at obs_vec.location_indices"""
    out_vals = [[state_vec.values[i][np.array(obs_vec.location_indices)].sum()]
                for i in range(state_vec.time_dim)]
    return vector.ObsVector(
                values=out_vals,
                times=state_vec.times,
                store_as_jax=state_vec.store_as_jax
                )

@pytest.fixture
def obsop_custom_h():
    """Defines ObsOp with custom h operator"""
    return obsop.ObsOp(
            h=custom_h
            )

@pytest.fixture
def obsvector_locindices_1():
    """Defines ObsVector with location_indices for other tests"""
    return vector.ObsVector(location_indices=[0])


@pytest.fixture
def obsvector_locindices_2():
    """Defines ObsVector with location_indices for other tests"""
    return vector.ObsVector(location_indices=[0, 1])


def test_defaults_singlesv(single_sv, obsop_default):
    """Test default settings on single timestep"""

    obs_vec = obsop_default.h(single_sv)

    assert obs_vec.obs_dims == np.array([2])
    assert obs_vec.time_dim == 1
    assert np.array_equal(obs_vec.times, np.array([0]))
    assert np.array_equal(obs_vec.location_indices, np.array([0, 1]))
    assert np.array_equal(obs_vec.values, np.array([[1., 2.]]))
    assert isinstance(obs_vec.values, np.ndarray)


def test_defaults_singlesv_jax(single_sv_jax, obsop_default):
    obs_vec = obsop_default.h(single_sv_jax)

    assert obs_vec.obs_dims == np.array([2])
    assert obs_vec.time_dim == 1
    assert np.array_equal(obs_vec.times, np.array([0]))
    assert np.array_equal(obs_vec.location_indices, np.array([0, 1]))
    assert np.array_equal(obs_vec.values,
                          jnp.array([[1., 2.]]))
    assert isinstance(obs_vec.values, jnp.ndarray)


def test_defaults_trajectory(sv_trajectory, obsop_default):
    """Test default settings on trajectory (multiple timesteps)"""

    obs_vec = obsop_default.h(sv_trajectory)

    assert np.array_equal(obs_vec.obs_dims, np.repeat(3, 8))
    assert obs_vec.time_dim == 8
    assert np.array_equal(obs_vec.times, np.arange(8))
    assert np.array_equal(obs_vec.location_indices, np.array([0, 1, 2]))
    assert np.array_equal(obs_vec.values,  np.arange(0, 24).reshape((8, 3)))
    assert isinstance(obs_vec.values, np.ndarray)


def test_man_locinds_singlesv(single_sv,
                              obsop_default,
                              obsvector_locindices_1):
    """Test manually specified indices on single timestep"""

    obs_vec = obsop_default.h(single_sv, obsvector_locindices_1)

    assert obs_vec.obs_dims == np.array([1])
    assert obs_vec.time_dim == 1
    assert np.array_equal(obs_vec.times, np.array([0]))
    assert np.array_equal(obs_vec.location_indices, np.array([0]))
    assert np.array_equal(obs_vec.values, np.array([[1]]))
    assert isinstance(obs_vec.values, np.ndarray)


def test_man_locinds_trajectory(sv_trajectory,
                                obsop_default,
                                obsvector_locindices_1):
    """Test manually specified indices on trajectory"""

    obs_vec = obsop_default.h(sv_trajectory, obsvector_locindices_1)

    assert np.array_equal(obs_vec.obs_dims, np.repeat(1, 8))
    assert obs_vec.time_dim == 8
    assert np.array_equal(obs_vec.times, np.arange(8))
    assert np.array_equal(obs_vec.location_indices, np.array([0]))
    assert np.array_equal(obs_vec.values,
                          np.arange(24, step=3, dtype='float').reshape(8, 1))
    assert isinstance(obs_vec.values, np.ndarray)


def test_custom_h_singlesv(single_sv, obsop_custom_h, obsvector_locindices_2):
    """Test default settings on single timestep"""

    obs_vec = obsop_custom_h.h(single_sv, obsvector_locindices_2)

    assert obs_vec.obs_dims == np.array([1])
    assert obs_vec.time_dim == 1
    assert np.array_equal(obs_vec.times, np.array([0]))
    assert np.array_equal(obs_vec.values, np.array([[3]]))
    assert isinstance(obs_vec.values, np.ndarray)


def test_custom_h_trajectory(sv_trajectory,
                             obsop_custom_h,
                             obsvector_locindices_2):
    """Test default settings on single timestep"""

    obs_vec = obsop_custom_h.h(sv_trajectory, obsvector_locindices_2)

    assert np.array_equal(obs_vec.obs_dims, np.repeat(1, 8))
    assert obs_vec.time_dim == 8
    assert np.array_equal(obs_vec.times, np.arange(8))
    assert np.array_equal(obs_vec.values,
                          np.array([[1], [7], [13], [19], [25], [31], [37],
                                    [43]]))
    assert isinstance(obs_vec.values, np.ndarray)
