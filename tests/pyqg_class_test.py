"""Tests for PYQG class (dabench.data.pyqg)"""

from dabench.data import PYQG
import jax.numpy as jnp
import pytest


@pytest.fixture
def pyqg():
    """Defines class PYQ object for rest of tests."""
    pyqg_obj = PYQG()
    pyqg_obj.generate(n_steps=1000)
    return pyqg_obj


def test_initialization(pyqg):
    """Tests the initialization size of class PYQG after generation."""
    assert pyqg.x0.shape == (2, 64, 64)


def test_variable_sizes(pyqg):
    """Test the variable sizes of class PYQG."""
    assert pyqg.system_dim == 8192
    assert pyqg.time_dim == 1000
    assert pyqg.original_dim == (2, 64, 64)


def test_to_original_dim(pyqg):
    assert pyqg.to_original_dim().shape == (pyqg.time_dim,) + pyqg.original_dim


def test_trajectories_equal(pyqg):
    """Tests if two trajectories are the same with same initial conditions."""
    pyqg2 = PYQG()
    pyqg2.generate(n_steps=1000)
    assert jnp.allclose(pyqg.values, pyqg2.values, rtol=1e-5, atol=0)


def test_trajectories_notequal_diffparams(pyqg):
    """Tests if two trajectories differ with different params."""
    pyqg2 = PYQG(rd=10000, H1=250)
    pyqg2.generate(n_steps=1000)
    assert not jnp.allclose(pyqg.values, pyqg2.values, rtol=1e-5, atol=0)


def test_trajectories_notequal_diffic(pyqg):
    """Tests if two trajectories differ with different initial conditions."""
    x0 = pyqg.x0 + 0.01
    pyqg2 = PYQG(x0=x0)
    pyqg2.generate(n_steps=1000)
    assert not jnp.allclose(pyqg.values, pyqg2.values, rtol=1e-5, atol=0)


def test_trajectory_changes(pyqg):
    """Tests that last time step in trajectory is different from initial state"""
    assert not jnp.allclose(pyqg.to_original_dim()[-1],
                            pyqg.x0)


def test_trajectory_shape(pyqg):
    """Tests output shape is (time_dim, system_dim)."""
    assert pyqg.values.shape == (pyqg.time_dim, pyqg.system_dim)
