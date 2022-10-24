"""Tests for DataBarotropic class (dabench.data.barotropic)"""

from dabench.data.barotropic import DataBarotropic
import jax.numpy as jnp
import pytest


@pytest.fixture
def barotropic():
    """Defines class DataPYQ object for rest of tests."""
    barotropic_obj = DataBarotropic()
    barotropic_obj.generate(n_steps=1000)
    return barotropic_obj


def test_initialization(barotropic):
    """Tests the initialization size of class DataBarotropic after generation."""
    assert barotropic.x0.shape == (1, 256, 256)


def test_variable_sizes(barotropic):
    """Test the variable sizes of class DataBarotropic."""
    assert barotropic.system_dim == 65536
    assert barotropic.time_dim == 1000
    assert barotropic.original_dim == (1, 256, 256)


def test_to_original_dim(barotropic):
    assert (barotropic.to_original_dim().shape ==
            (barotropic.time_dim,) + barotropic.original_dim)


def test_trajectories_equal(barotropic):
    """Tests if two trajectories are the same with same initial conditions."""
    barotropic2 = DataBarotropic()
    barotropic2.generate(n_steps=1000)
    assert jnp.allclose(barotropic.values, barotropic2.values, rtol=1e-5,
                        atol=0)


def test_trajectories_notequal_diffparams(barotropic):
    """Tests if two trajectories differ with different params."""
    barotropic2 = DataBarotropic(rd=10000, H=250)
    barotropic2.generate(n_steps=1000)
    assert not jnp.allclose(barotropic.values, barotropic2.values, rtol=1e-5,
                            atol=0)


def test_trajectories_notequal_diffic(barotropic):
    """Tests if two trajectories differ with different initial conditions."""
    x0 = barotropic.x0 + 0.01
    barotropic2 = DataBarotropic(x0=x0)
    barotropic2.generate(n_steps=1000)
    assert not jnp.allclose(barotropic.values, barotropic2.values, rtol=1e-5,
                            atol=0)


def test_trajectory_changes(barotropic):
    """Tests that last time step in trajectory is different from initial state"""
    assert not jnp.allclose(barotropic.to_original_dim()[-1],
                            barotropic.x0)


def test_trajectory_shape(barotropic):
    """Tests output shape is (time_dim, system_dim)."""
    assert barotropic.values.shape == (barotropic.time_dim, barotropic.system_dim)
