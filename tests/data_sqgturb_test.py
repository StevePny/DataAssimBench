
"""Tests for SQGTurb class (dabench.data.sqgturb)"""

import jax.numpy as jnp
import jax.random as jrand
import pytest

from dabench.data import SQGTurb


key = jrand.PRNGKey(42)


@pytest.fixture
def sqgturb():
    """Defines class SQGTurb object for rest of tests."""
    N = 96

    # create random noise
    pv = 100*jrand.normal(key, shape=(2, N, N), dtype=jnp.float32)
    # add isolated blob on lid
    nexp = 20
    x = jnp.arange(0, 2.*jnp.pi, 2.*jnp.pi/N)
    y = jnp.arange(0., 2.*jnp.pi, 2.*jnp.pi/N)
    x, y = jnp.meshgrid(x, y)
    x = x.astype(jnp.float32)
    y = y.astype(jnp.float32)
    pv = pv.at[1].set(pv[1]+2000.*(jnp.sin(x/2)**(2*nexp)*jnp.sin(y)**nexp))
    # remove area mean from each level.
    for k in range(2):
        pv = pv.at[k].set(pv[k] - pv[k].mean())

    # initialize qg model instance
    return SQGTurb(pv=pv)


def test_variable_sizes(sqgturb):
    """Test the variable sizes of class SQGTurb."""
    n_steps = 10
    sqgturb.generate(n_steps=n_steps)

    assert sqgturb.system_dim == 9408
    assert sqgturb.time_dim == n_steps+1
    assert sqgturb.values.shape == (sqgturb.time_dim, sqgturb.system_dim)
