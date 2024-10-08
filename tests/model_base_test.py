"""Tests for Model class (dabench.model._model)"""

import pytest
import numpy as np
import xarray as xr

from dabench import model, data, vector


class L63Model(model.Model):
    """Defines model wrapper for Lorenz63 to test forecasting."""
    def forecast(self, state_vec):
        new_vec = self.model_obj.generate(x0=state_vec['x'].values, n_steps=2)

        return new_vec.isel(time=-1), new_vec


def test_l63_model_forecast():
    """Test forecast using wrapped Lorenz63 data generator"""
    l63_model = L63Model(model_obj=data.Lorenz63())

    state_vec = xr.Dataset(
        {'x':('index', l63_model.model_obj.x0)}
    )
    new_state_vec, _ = l63_model.forecast(state_vec)
    assert not np.allclose(new_state_vec['x'].values, state_vec['x'].values)
    assert np.allclose(state_vec['x'].values, np.array([-10., -15., 21.3]))
    assert np.allclose(new_state_vec['x'].values,
                        np.array([-10.499835, -15.48437, 22.282299]))

