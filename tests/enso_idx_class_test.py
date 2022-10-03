"""Tests for DataENSOIDX class (dabench.data.enso_idx)"""

from dabench.data.enso_idx import DataENSOIDX
import jax.numpy as jnp
import pytest


def test_initialization():
    """Test initialization of enso idx data"""
    file_list = {'wnd': ['zwnd200'],
                 'slp': ['darwin']}
    vtype = {'wnd': ['ori', 'ano'],
             'slp': ['ori', 'std']}

    idx = DataENSOIDX(file_list, vtype)

    assert idx.values.shape[1] == 4
    assert idx.values.shape[0] == 524
    assert idx.values[0, 0] == 15.5
