"""Tests for QGS class (dabench.data.qgs)"""

import numpy as np
import pytest

from dabench.data import QGS


@pytest.fixture(scope='module')
def qgs_traj_tuple():
    """Defines class QGS object for rest of tests."""
    qgs_obj = QGS()
    traj = qgs_obj.generate(n_steps=10)
    return qgs_obj, traj


def test_variable_sizes(qgs_traj_tuple):
    """Test the variable sizes of class QGS."""
    qgs, traj = qgs_traj_tuple
    assert qgs.x0.shape == (36,)
    assert qgs.system_dim == 36
    assert traj.system_dim == 36
    assert traj.sizes['time'] == 10


def test_trajectories_equal(qgs_traj_tuple):
    """Tests if two trajectories are the same with same initial conditions."""
    qgs, traj = qgs_traj_tuple
    qgs2 = QGS()
    traj2 = qgs2.generate(n_steps=10)
    assert np.allclose(traj.dab.flatten().values, traj2.dab.flatten().values,
                       rtol=1e-5, atol=0)


def test_trajectories_notequal_diffparams(qgs_traj_tuple):
    """Tests if two trajectories differ with different params."""
    qgs, traj = qgs_traj_tuple = qgs_traj_tuple

    qgs_params = qgs.model_params
    qgs_params.set_params({'kd': 0.0390, 'kdp': 0.0390, 'n': 1.4, 'h': 126.5})

    qgs2 = QGS(model_params=qgs_params)
    traj2 = qgs2.generate(n_steps=10)
    assert not np.allclose(traj.dab.flatten().values, traj2.dab.flatten().values,
                           rtol=1e-5, atol=0)


def test_trajectories_notequal_diffic(qgs_traj_tuple):
    """Tests if two trajectories differ with different initial conditions."""
    qgs, traj = qgs_traj_tuple
    new_x0 = qgs.x0 + 0.01
    qgs2 = QGS(x0=new_x0)
    traj2 = qgs2.generate(n_steps=10)
    assert not np.allclose(traj.dab.flatten().values, traj2.dab.flatten().values,
                           rtol=1e-5, atol=0)


def test_trajectory_changes(qgs_traj_tuple):
    """Tests that last time step in trajectory is different from initial state"""
    qgs, traj = qgs_traj_tuple

    assert np.allclose(traj.dab.flatten().values[0],
                           qgs.x0)
    assert not np.allclose(traj.dab.flatten().values[-1],
                           qgs.x0)


def test_trajectory_shape(qgs_traj_tuple):
    """Tests output shape is (time_dim, system_dim)."""
    qgs, traj = qgs_traj_tuple
    assert traj.to_array().shape == (1, traj.sizes['time'], qgs.system_dim)
