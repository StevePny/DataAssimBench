"""Base class for Model object

Wrapper around user-specified model. The user must define a sub-class that
    inherits from dabench.model.Model, with an forecast() method.

Optional, for the compiled-window path of the matrix-free 4D-Var cyclers
(``Var4DOperator`` / ``Var4DOperatorWC`` ``cycle(model_inputs=...,
compile_once=True)``): a ``step_fn()`` method returning the one-step map on a
flat state, either ``F(x)`` (time-independent) or ``F(x, t, u)`` with ``t`` the
model time of the step and ``u`` that step's row of the ``model_inputs``
pytree (None when none are given).  Time-varying data (forcing, boundary
frames) then reach the step as data rather than as attributes of the model,
so the compiled window does not depend on them and is reused from cycle to
cycle.  ``forecast(state_vec, n_steps)`` keeps its contract; a model may
additionally accept ``forecast(state_vec, n_steps, t0=None, inputs=None)``
for its own use (the cyclers' original path never passes them).
"""

from typing import Any
import xarray as xr

class Model():
    """Base for Model objects

    Args:
        system_dim: system dimension
        delta_t: the timestep of the model (assumed uniform)
        model_obj: underlying model object, e.g. pytorch neural network.
    """
    def __init__(self,
                 system_dim: int | None = None,
                 delta_t: int | None = None,
                 model_obj: Any | None = None
                 ):

        self.system_dim = system_dim
        self.delta_t = delta_t
        self.model_obj = model_obj

        # Check that forecast is a defined method
        forecast_method = getattr(self, 'forecast', None)
        if not callable(forecast_method):
            raise ValueError('Model object does not have a defined forecast() '
                             'method.')

    def _default_forecast(self,
                          state_vec: xr.Dataset,
                          timesteps: int = 1,
                          other_inputs: Any = None):
        """Default method for forecasting"""
        new_state_vec = state_vec
        for i in range(timesteps):
            new_state_vec = self.model_obj.predict(new_state_vec)

        return new_state_vec
