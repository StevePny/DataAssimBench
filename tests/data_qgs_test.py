"""Tests for QGS class (dabench.data.qgs)"""

import numpy as np
import pytest

from dabench.data import QGS


@pytest.fixture(scope='module')
def qgs():
    """Defines class QGS object for rest of tests."""
    qgs_obj = QGS()
    qgs_obj.generate(n_steps=1000)
    return qgs_obj


def test_initialization(qgs):
    """Tests the initialization size of class QGS after generation."""
    assert qgs.x0.shape == (36,)


def test_variable_sizes(qgs):
    """Test the variable sizes of class QGS."""
    assert qgs.system_dim == 36
    assert qgs.time_dim == 1000
    assert qgs.original_dim == (36,)


def test_trajectories_equal(qgs):
    """Tests if two trajectories are the same with same initial conditions."""
    qgs2 = QGS()
    qgs2.generate(n_steps=1000)
    assert np.allclose(qgs.values, qgs2.values, rtol=1e-5, atol=0)


def test_trajectories_notequal_diffparams(qgs):
    """Tests if two trajectories differ with different params."""
    qgs_params = qgs.model_params
    qgs_params.set_params({'kd': 0.0390, 'kdp': 0.0390, 'n': 1.4, 'h': 126.5})

    qgs2 = QGS(model_params=qgs_params)
    qgs2.generate(n_steps=1000)
    assert not np.allclose(qgs.values, qgs2.values, rtol=1e-5, atol=0)


def test_trajectories_notequal_diffic(qgs):
    """Tests if two trajectories differ with different initial conditions."""
    new_x0 = qgs.x0 + 0.01
    qgs2 = QGS(x0=new_x0)
    qgs2.generate(n_steps=1000)
    assert not np.allclose(qgs.values, qgs2.values, rtol=1e-5, atol=0)


def test_trajectory_changes(qgs):
    """Tests that last time step in trajectory is different from initial state"""
    assert not np.allclose(qgs.values[-1],
                           qgs.x0)


def test_trajectory_shape(qgs):
    """Tests output shape is (time_dim, system_dim)."""
    assert qgs.values.shape == (qgs.time_dim, qgs.system_dim)
