"""Tests for deterministic metrics (dabench.metrics._deterministic)"""

from dabench import metrics

import numpy as np
import pytest


@pytest.fixture
def pred_flat():
    np_rng = np.random.default_rng(10)
    return np_rng.normal(size=10)

@pytest.fixture
def targ_flat():
    np_rng = np.random.default_rng(11)
    return np_rng.normal(size=10)


def test_cov(pred_flat, targ_flat):
    """Test covariance calculation"""

    cov = metrics._utils._cov(pred_flat, targ_flat)
    assert np.isclose(cov, -0.04873383)

def test_mae(pred_flat, targ_flat):
    """Test covariance calculation"""

    mae = metrics.mae(pred_flat, targ_flat)
    assert np.isclose(mae, 0.9564944069125414)

def test_mse(pred_flat, targ_flat):
    """Test covariance calculation"""

    mse = metrics.mse(pred_flat, targ_flat)
    assert np.isclose(mse, 1.3634841377574962)

def test_rmse(pred_flat, targ_flat):
    """Test covariance calculation"""

    rmse = metrics.rmse(pred_flat, targ_flat)
    assert np.isclose(rmse, 1.1676832351958712)


