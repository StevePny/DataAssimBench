"""Tests for PyQG class (dabench.data.pyqg)"""

import numpy as np
import pytest

from dabench.data import PyQG


@pytest.fixture(scope='module')
def pyqg():
    """Defines class PYQ object for rest of tests."""
    pyqg_obj = PyQG()
    pyqg_obj.generate(n_steps=1000)
    return pyqg_obj


def test_initialization(pyqg):
    """Tests the initialization size of class PyQG after generation."""
    assert pyqg.x0.shape == (8192,)
    assert pyqg.x0_gridded.shape == (2, 64, 64)


def test_variable_sizes(pyqg):
    """Test the variable sizes of class PyQG."""
    assert pyqg.system_dim == 8192
    assert pyqg.time_dim == 1000
    assert pyqg.original_dim == (2, 64, 64)


def test_to_original_dim(pyqg):
    assert pyqg._to_original_dim().shape == (pyqg.time_dim,) + pyqg.original_dim


def test_trajectories_equal(pyqg):
    """Tests if two trajectories are the same with same initial conditions."""
    pyqg2 = PyQG()
    pyqg2.generate(n_steps=1000)
    assert np.allclose(pyqg.values, pyqg2.values, rtol=1e-5, atol=0)


def test_trajectories_notequal_diffparams(pyqg):
    """Tests if two trajectories differ with different params."""
    pyqg2 = PyQG(rd=10000, H1=250)
    pyqg2.generate(n_steps=1000)
    assert not np.allclose(pyqg.values, pyqg2.values, rtol=1e-5, atol=0)


def test_trajectories_notequal_diffic(pyqg):
    """Tests if two trajectories differ with different initial conditions."""
    new_x0 = pyqg.x0_gridded + 0.01
    pyqg2 = PyQG(x0=new_x0)
    pyqg2.generate(n_steps=1000)
    assert not np.allclose(pyqg.values, pyqg2.values, rtol=1e-5, atol=0)


def test_trajectory_changes(pyqg):
    """Tests that last time step in trajectory is different from initial state"""
    assert not np.allclose(pyqg.values_gridded[-1],
                           pyqg.x0_gridded)


def test_trajectory_shape(pyqg):
    """Tests output shape is (time_dim, system_dim)."""
    assert pyqg.values.shape == (pyqg.time_dim, pyqg.system_dim)
