"""Tests for Barotropic class (dabench.data.barotropic)"""

from dabench.data import Barotropic
import numpy as np
import pytest


@pytest.fixture(scope='module')
def barotropic():
    """Defines class Barotropic object for rest of tests."""
    barotropic_obj = Barotropic()
    barotropic_obj.generate(n_steps=1000)
    return barotropic_obj


def test_initialization(barotropic):
    """Tests the initialization size of class Barotropic after generation."""
    assert barotropic.x0.shape == (65536,)
    assert barotropic.x0_gridded.shape == (1, 256, 256)


def test_variable_sizes(barotropic):
    """Test the variable sizes of class Barotropic."""
    assert barotropic.system_dim == 65536
    assert barotropic.time_dim == 1000
    assert barotropic.original_dim == (1, 256, 256)


def test_to_original_dim(barotropic):
    assert (barotropic._to_original_dim().shape ==
            (barotropic.time_dim,) + barotropic.original_dim)


def test_trajectories_equal(barotropic):
    """Tests if two trajectories are the same with same initial conditions."""
    barotropic2 = Barotropic()
    barotropic2.generate(n_steps=1000)
    assert np.allclose(barotropic.values, barotropic2.values, rtol=1e-5,
                       atol=0)


def test_trajectories_notequal_diffparams(barotropic):
    """Tests if two trajectories differ with different params."""
    barotropic2 = Barotropic(rd=10000, H=250)
    barotropic2.generate(n_steps=1000)
    assert not np.allclose(barotropic.values, barotropic2.values, rtol=1e-5,
                           atol=0)


def test_trajectories_notequal_diffic(barotropic):
    """Tests if two trajectories differ with different initial conditions."""
    new_x0 = barotropic.x0_gridded + 0.01
    barotropic2 = Barotropic(x0=new_x0)
    barotropic2.generate(n_steps=1000)
    assert not np.allclose(barotropic.values, barotropic2.values, rtol=1e-5,
                           atol=0)


def test_trajectory_changes(barotropic):
    """Tests that last time step in trajectory is different from initial state"""
    assert not np.allclose(barotropic.values_gridded[-1],
                           barotropic.x0_gridded)


def test_trajectory_shape(barotropic):
    """Tests output shape is (time_dim, system_dim)."""
    assert barotropic.values.shape == (barotropic.time_dim, barotropic.system_dim)
