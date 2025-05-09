"""Tests for Lorenz96 class (from dabench.data.lorenz96)"""

import numpy as np
import pytest

from dabench.data import Lorenz96


@pytest.fixture
def lorenz96():
    """Defines class Lorenz96 object for rest of tests."""
    params = {'system_dim': 5,
              'forcing_term': 8.0,
              'delta_t': 0.01
              }
    return Lorenz96(**params)


@pytest.fixture
def lorenz96_lyaps(lorenz96):
    """Calculates Lyapunov Exponent values for tests."""
    return lorenz96.calc_lyapunov_exponents_series(total_time=20)


def test_initialization():
    """Tests the initialized size of class Lorenz96 after generation."""
    for i in range(1, 10):
        params = {'system_dim': 5*i}
        lorenz96_1 = Lorenz96(**params)

        assert lorenz96_1.system_dim == 5*i


def test_variable_sizes(lorenz96):
    """Test the variable sizes of class Lorenz96."""
    runtime = 1
    traj = lorenz96.generate(t_final=runtime)

    assert traj.system_dim == 5
    assert traj.sizes['time'] == runtime/lorenz96.delta_t
    assert traj.delta_t == lorenz96.delta_t


def test_trajectory_shape():
    """Tests output shape is (runtime, sys_dim)."""
    params = {'system_dim': 20,
              'forcing_term': 8.0,
              'delta_t': 0.01
              }
    lorenz96_1 = Lorenz96(**params)
    runtime = 1
    traj_1 = lorenz96_1.generate(t_final=runtime)

    assert traj_1.to_array().shape == (1,
                                       int(runtime/lorenz96_1.delta_t),
                                       lorenz96_1.system_dim)


def test_trajectories_equal():
    """Tests if two trajectories are the same with same initial conditions."""
    params = {'system_dim': 6,
              'forcing_term': 10.0,
              'delta_t': 0.001
              }
    lorenz96_1 = Lorenz96(**params)
    lorenz96_2 = Lorenz96(**params)
    runtime = 1
    traj1 = lorenz96_1.generate(t_final=runtime)
    traj2 = lorenz96_2.generate(t_final=runtime)

    assert np.allclose(traj1.to_array().data, traj2.to_array().data, rtol=1e-5,
                       atol=0)


def test_trajectories_notequal():
    """Tests if two trajectories are the same with same initial conditions."""
    params = {'system_dim': 6,
              'forcing_term': 8.0,
              'delta_t': 0.01
              }
    lorenz96_1 = Lorenz96(**params)
    lorenz96_2 = Lorenz96(x0=np.array([6.20995768, 6.24066944,
                                       4.27604607, 4.25271592,
                                       -3.11392061, 3.52697510]),
                          **params)
    runtime = 1
    traj1 = lorenz96_1.generate(t_final=runtime)
    traj2 = lorenz96_2.generate(t_final=runtime)

    assert not np.allclose(traj1.to_array().data, traj2.to_array().data, rtol=1e-5,
                           atol=0)


def test_trajectory_changes(lorenz96):
    """Tests that last time step in trajectory is different from initial"""
    runtime = 1
    initial_conditions = np.array([7.97355787, 7.97897913, 8.00370696,
                                   7.98444298, 7.97446945])
    traj1 = lorenz96.generate(t_final=runtime, x0 = initial_conditions)

    assert not np.allclose(traj1.isel(time=-1)['x'].data, initial_conditions)


def test_generate_saved_results():
    """Tests the Lorenz96 data generation against below true array results."""
    # Set initial state to equilibrium with small perturbation to third value
    x0 = 8.0 * np.ones(5)
    x0[2] = x0[2]+0.01
    params = {'system_dim': 5,
              'forcing_term': 8.0,
              'delta_t': 0.001,
              'x0': x0
              }

    # Generate data
    lorenz96_1 = Lorenz96(**params)
    traj = lorenz96_1.generate(t_final=10, x0=x0)

    # Previously generated results with these params and initial conditions
    y_true = np.array(
            [[ 8.        ,  8.        ,  8.01      ,  8.        ,  8.        ],
             [-0.98989847,  3.37430623, 11.26913436, 12.84401465, -0.72462902],
             [ 2.61054548, -0.07543557,  0.32725793, 11.42165675,  1.87837841],
             [ 3.20376924,  6.83403523, -2.63507472,  3.05415108,  2.56797523],
             [ 0.20204786,  1.26443727,  8.39982716, -2.45623576,  0.39051976],
             [-5.27169559,  0.97322104,  2.41156636,  6.96395228, -1.70612383],
             [ 4.26726853,  6.23099434, -1.22264959,  2.29152838,  2.84244614],
             [-0.25361909,  0.92942401,  8.63986765, -1.45411412, -0.89528154],
             [-5.27402622, -1.10262156, -0.53270967,  7.58069989,  2.84768712],
             [ 4.73878926,  6.27758562, -2.10858776,  3.26147038,  3.168998  ]])

    y_simulated = traj['x'].values[::1000]
    assert y_simulated.shape == y_true.shape
    assert np.allclose(y_simulated, y_true, rtol=0.01, atol=0)
    assert np.allclose(np.sum(y_simulated), np.sum(y_true), rtol=1e-3,
                       atol=0)


def test_lyapunov_exponents(lorenz96, lorenz96_lyaps):
    """Tests that shape of lyapunov exponents is same as system_dim"""
    LE = lorenz96_lyaps[-1]
    assert len(LE) == lorenz96.system_dim


def test_lyapunov_exponents_series(lorenz96, lorenz96_lyaps):
    """Tests shape of lyapunov exponents series and value of last timestep"""
    LE = lorenz96.calc_lyapunov_exponents_final(total_time=20)
    assert lorenz96_lyaps.shape == (20 - 1, lorenz96.system_dim)
    assert np.all(LE == lorenz96_lyaps[-1])


def test_lyapunov_exponents_values(lorenz96_lyaps):
    """Tests that Lorenz96 lyapunov exponents are close to known values."""
    LE = lorenz96_lyaps[-1]
    known_LE = np.array([0.55787979, -0.06681031, -0.42554347, -1.30596795,
                         -3.70955825])
    assert np.allclose(known_LE, LE,  rtol=0.05, atol=0.01)
