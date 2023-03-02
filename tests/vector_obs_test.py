"""Tests for StateVector class (dabench.vector._state_vector)"""

import pytest
import numpy as np
import jax.numpy as jnp

from dabench.vector import ObsVector


@pytest.fixture
def obsvec():
    """Defines Basic ObsVector"""
    params = {'num_obs': 5,
              'values': np.array([1, 2, 3, 4, 5]),
              'times': np.array([0, 5, 10, 15, 20]),
              'errors': np.array([0.1, -0.15, 0.05, 0.2, -0.2]),
              'location_indices':  np.array([[0, 0], [0, 1], [1, 1], [2, 2],
                                             [5, 3]])
              }

    return ObsVector(**params)


@pytest.fixture
def obsvec_jax():
    """Defines Basic ObsVector with Jax backend"""
    params = {'num_obs': 5,
              'values': np.array([1, 2, 3, 4, 5]),
              'times': np.array([0, 5, 10, 15, 20]),
              'errors': np.array([0.1, -0.15, 0.05, 0.2, -0.2]),
              'store_as_jax': True
              }

    return ObsVector(**params)


@pytest.fixture
def obsvec_dt():
    """Defines Basic ObsVector with Datetime times"""
    params = {'num_obs': 4,
              'values': np.array([10, 11, 50, 25]),
              'times': np.array(['2005-01-01', '2005-02-01',
                                 '2006-02-03', '2005-06-01'],
                                dtype='datetime64'),
              'errors': np.array([0.1, -0.15, 0.05, 0.2])
              }

    return ObsVector(**params)


def test_init(obsvec):
    """Test the initialization of ObsVector"""

    assert obsvec.num_obs == 5
    assert obsvec.times.shape == (5,)
    assert np.array_equal(obsvec.obs_dims, np.repeat(1, 5))
    assert np.array_equal(obsvec.values, np.array([1, 2, 3, 4, 5]))
    assert isinstance(obsvec.values, np.ndarray)


def test_init_jax(obsvec_jax):
    """Test the initialization of ObsVector with Jax backend"""

    assert obsvec_jax.num_obs == 5
    assert obsvec_jax.times.shape == (5,)
    assert np.array_equal(obsvec_jax.obs_dims, np.repeat(1, 5))
    assert jnp.array_equal(obsvec_jax.values, jnp.array([1, 2, 3, 4, 5]))
    assert isinstance(obsvec_jax.values, jnp.ndarray)


def test_init_datetime(obsvec_dt):
    """Test initialization of StateVector trajectory (time_dim==8)"""

    assert obsvec_dt.num_obs == 4
    assert obsvec_dt.times.shape == (4,)
    assert np.array_equal(obsvec_dt.obs_dims, np.repeat(1, 4))
    assert np.array_equal(obsvec_dt.values,  np.array([10, 11, 50, 25]))


def test_set_values():
    """Test manually setting vector values"""

    test_vec = ObsVector()

    new_values = np.arange(15)
    test_vec.values = new_values

    assert np.array_equal(test_vec.values, new_values)


def test_set_values_jax():
    """Test manually setting vector values with jax backend"""

    test_vec = ObsVector(store_as_jax=True)

    new_values = np.arange(15)
    test_vec.values = new_values

    assert jnp.array_equal(test_vec.values, new_values)
    assert isinstance(test_vec.values, jnp.ndarray)


def test_time_filter_int(obsvec):
    """Tests filter_times method"""

    newvec = obsvec.filter_times(0, 10)

    assert newvec.values.shape != obsvec.values.shape
    assert newvec.num_obs == 3
    assert np.array_equal(newvec.obs_dims, np.repeat(1, 3))
    assert np.array_equal(newvec.values, np.array([1, 2, 3]))
    assert np.array_equal(newvec.times, np.array([0, 5, 10]))


def test_time_filter_int_exc(obsvec):
    """Tests filter_times method"""

    newvec = obsvec.filter_times(0, 10, inclusive=False)

    assert newvec.values.shape != obsvec.values.shape
    assert newvec.num_obs == 1
    assert np.array_equal(newvec.obs_dims, np.repeat(1, 1))
    assert np.array_equal(newvec.values, np.array([2]))
    assert np.array_equal(newvec.times, np.array([5]))


def test_time_filter_dt(obsvec_dt):
    """Tests filter_times method"""

    newvec = obsvec_dt.filter_times(np.datetime64('2005-02-01'),
                                    np.datetime64('2006-01-01')
                                    )

    assert newvec.values.shape != obsvec_dt.values.shape
    assert newvec.num_obs == 2
    assert np.array_equal(newvec.obs_dims, np.repeat(1, 2))
    assert np.array_equal(newvec.values, np.array([11, 25]))
