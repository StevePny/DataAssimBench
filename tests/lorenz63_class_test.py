"""Tests for data class (dabench.data.data)"""

from dabench.data import data
from dabench.data.lorenz63 import DataLorenz63
import jax.numpy as jnp


def test_initialization():
    """Test the initialization size of class_data_Lorenz63 after generation."""
    params = {'system_dimension': 3}
    lorenz = DataLorenz63(**params)
    lorenz.generate(t_final=1)

    assert len(lorenz.x0) == 3


def test_variable_sizes():
    """Test the variable sizes of class_data_Lorenz63."""
    params = {'system_dimension': 3}
    lorenz = DataLorenz63(**params)
    runtime = 1
    lorenz.generate(t_final=runtime)

    assert lorenz.system_dim == 3
    assert lorenz.time_dim == runtime/lorenz.dt


def test_trajectories_equal():
    """Test if two trajectories are the same with same initial conditions."""
    params = {'system_dimension': 3}
    lorenz = DataLorenz63(**params)
    lorenz2 = DataLorenz63(**params)
    runtime = 1
    lorenz.generate(t_final=runtime)
    lorenz2.generate(t_final=runtime)

    assert jnp.allclose(lorenz.values, lorenz2.values, rtol=1e-5, atol=0)


def test_trajectory_shape():
    """Test output shape is (sys_dim, runtime)."""
    params = {'system_dimension': 3}
    lorenz = DataLorenz63(**params)
    # Typical Lorenz63 setup
    runtime = 1
    sys_dim = 3 
    lorenz.generate(t_final=runtime)

    assert lorenz.values.shape == (sys_dim, int(runtime/lorenz.dt))
