"""Tests for Model class (dabench.model._model)"""

import pytest
import numpy as np

from dabench import model, data, vector


class L63Model(model.Model):
    """Defines model wrapper for Lorenz63 to test forecasting."""
    def forecast(self, state_vec):
        self.model_obj.generate(x0=state_vec.values, n_steps=2)
        new_vals = self.model_obj.values[-1]

        new_vec = vector.StateVector(values=new_vals)

        return new_vec


def test_l63_model_forecast():
    """Test forecast using wrapped Lorenz63 data generator"""
    l63_model = L63Model(model_obj=data.Lorenz63())

    state_vec = vector.StateVector(values=l63_model.model_obj.x0)
    new_state_vec = l63_model.forecast(state_vec)
    assert not np.allclose(new_state_vec.values, state_vec.values)
    assert np.allclose(state_vec.values, np.array([-10., -15., 21.3]))
    assert np.allclose(new_state_vec.values,
                        np.array([-10.499835, -15.48437, 22.282299]))

