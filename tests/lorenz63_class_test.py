"""Tests for DataLorenz63 class (dabench.data.lorenz63)"""

from dabench.data.lorenz63 import DataLorenz63
import jax.numpy as jnp
import pytest


@pytest.fixture
def lorenz():
    """Defines class DataLorenz63 object for rest of tests."""
    params = {'system_dimension': 3}
    return DataLorenz63(**params)


@pytest.fixture
def lorenz_lyaps(lorenz):
    """Calculates Lyapunov Exponent values for tests."""
    return lorenz.calc_lyapunov_exponents_series()


def test_initialization(lorenz):
    """Tests the initialization size of class DataLorenz63 after generation."""
    lorenz.generate(t_final=1)

    assert len(lorenz.x0) == 3


def test_variable_sizes(lorenz):
    """Test the variable sizes of class DataLorenz63."""
    runtime = 1
    lorenz.generate(t_final=runtime)

    assert lorenz.system_dim == 3
    assert lorenz.time_dim == runtime/lorenz.delta_t


def test_trajectories_equal(lorenz):
    """Tests if two trajectories are the same with same initial conditions."""
    params = {'system_dimension': 3}
    lorenz2 = DataLorenz63(**params)
    runtime = 1
    lorenz.generate(t_final=runtime)
    lorenz2.generate(t_final=runtime)

    assert jnp.allclose(lorenz.values, lorenz2.values, rtol=1e-5, atol=0)


def test_trajectories_notequal(lorenz):
    """Tests if two trajectories differ with different initial conditions."""
    params = {'system_dimension': 3}
    lorenz2 = DataLorenz63(**params)
    runtime = 1
    lorenz.generate(t_final=runtime)
    lorenz2.generate(t_final=runtime,
                     x0=jnp.array([-2.2, -2.2, 19.1]))

    assert not jnp.allclose(lorenz.values, lorenz2.values, rtol=1e-5, atol=0)


def test_trajectory_changes(lorenz):
    """Tests that last time step in trajectory is different from initial state"""
    runtime = 1
    lorenz.generate(t_final=runtime,
                    x0=jnp.array([-2.2, -2.2, 19.1]))

    assert not jnp.allclose(lorenz.values[-1],  jnp.array([-2.2, -2.2, 19.1]))


def test_trajectory_shape(lorenz):
    """Tests output shape is (time_dim, system_dim)."""
    # Typical Lorenz63 setup
    runtime = 1
    lorenz.generate(t_final=runtime)

    assert lorenz.values.shape == (lorenz.time_dim, lorenz.system_dim)


def test_return_tlm_shape(lorenz):
    """Tests that tlm shape is (time_dim, system_dim, system_dim)"""
    runtime = 1
    tlm = lorenz.generate(t_final=runtime, return_tlm=True)
    assert tlm.shape == (lorenz.time_dim, lorenz.system_dim,
                         lorenz.system_dim)


def test_lyapunov_exponents(lorenz, lorenz_lyaps):
    """Tests that shape of lyapunov exponents is same as system_dim"""
    LE = lorenz_lyaps[-1]
    assert len(LE) == lorenz.system_dim


def test_lyapunov_exponents_series(lorenz, lorenz_lyaps):
    """Tests shape of lyapunov exponents series and value of last timestep"""
    LE = lorenz.calc_lyapunov_exponents_final()
    assert lorenz_lyaps.shape == (150 - 1, lorenz.system_dim)
    assert jnp.all(LE == lorenz_lyaps[-1])


def test_lyapunov_exponents_values(lorenz_lyaps):
    """Tests that Lorenz63 lyapunov exponents are close to known values.

    Note:
        Values from https://sprott.physics.wisc.edu/chaos/lorenzle.htm
    """
    LE = lorenz_lyaps[-1]
    known_LE = jnp.array([0.906, 0, -14.572])
    assert jnp.allclose(known_LE, LE,  rtol=0.05, atol=0.01)

