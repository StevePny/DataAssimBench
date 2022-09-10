"""Tests for data class (dabench.data.data)"""

from dabench.data import data
from dabench.data.lorenz63 import DataLorenz63
import jax.numpy as jnp
import pytest


@pytest.fixture
def lorenz():
    """Define Lorenz data object for rest of tests."""
    params = {'system_dimension': 3}
    return DataLorenz63(**params)


def test_initialization(lorenz):
    """Test the initialization size of class_data_Lorenz63 after generation."""
    lorenz.generate(t_final=1)

    assert len(lorenz.x0) == 3


def test_variable_sizes(lorenz):
    """Test the variable sizes of class_data_Lorenz63."""
    runtime = 1
    lorenz.generate(t_final=runtime)

    assert lorenz.system_dim == 3
    assert lorenz.time_dim == runtime/lorenz.dt


def test_trajectories_equal(lorenz):
    """Test if two trajectories are the same with same initial conditions."""
    params = {'system_dimension': 3}
    lorenz2 = DataLorenz63(**params)
    runtime = 1
    lorenz.generate(t_final=runtime)
    lorenz2.generate(t_final=runtime)

    assert jnp.allclose(lorenz.values, lorenz2.values, rtol=1e-5, atol=0)


def test_trajectories_notequal(lorenz):
    """Test if two trajectories are the different with different initial conditions."""
    params = {'system_dimension': 3}
    lorenz2 = DataLorenz63(**params)
    runtime = 1
    lorenz.generate(t_final=runtime)
    lorenz2.generate(t_final=runtime,
                     x0=jnp.array([-2.2, -2.2, 19.1]))

    assert not jnp.allclose(lorenz.values, lorenz2.values, rtol=1e-5, atol=0)


def test_trajectory_changes(lorenz):
    """Test that last time step in trajectory is different from initial state"""
    runtime = 1
    lorenz.generate(t_final=runtime,
                    x0=jnp.array([-2.2, -2.2, 19.1]))

    assert not jnp.allclose(lorenz.values[-1],  jnp.array([-2.2, -2.2, 19.1]))


def test_trajectory_shape(lorenz):
    """Test output shape is (runtime, sys_dim)."""
    # Typical Lorenz63 setup
    runtime = 1
    sys_dim = 3
    lorenz.generate(t_final=runtime)

    assert lorenz.values.shape == (int(runtime/lorenz.dt), sys_dim)
