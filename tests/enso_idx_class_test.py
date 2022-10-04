"""Tests for DataENSOIDX class (dabench.data.enso_idx)"""

from dabench.data.enso_idx import DataENSOIDX
import jax.numpy as jnp
import pytest


def test_initialization():
    """Test initialization of enso idx data"""
    file_dict = {'wnd': ['zwnd200'],
                 'slp': ['darwin']}
    var_types = {'wnd': ['ori', 'ano'],
                 'slp': ['ori', 'std']}

    idx = DataENSOIDX(file_dict, var_types)

    assert idx.values[0, 0] == 15.5
    assert idx.values.shape[1] == 4
