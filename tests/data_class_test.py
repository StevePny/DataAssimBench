"""Tests for data class (dabench.data.data)"""

from dabench.data import data
import jax.numpy as jnp


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
