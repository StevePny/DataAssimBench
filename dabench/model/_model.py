"""Base class for Model object

Wrapper around user-specified model. The user must define a sub-class that
    inherits from dabench.model.Model, with an forecast() method.
"""

from typing import Any
import xarray as xr

class Model():
    """Base class for Model object

    Attributes:
        system_dim (int): system dimension
        time_dim (int): total time steps
        delta_t (float): the timestep of the model (assumed uniform)
        model_obj (obj): underlying model object, e.g. pytorch neural network.
    """
    def __init__(self,
                 system_dim: int | None = None,
                 time_dim: int | None = None,
                 delta_t: int | None = None,
                 model_obj: int | None = None
                 ):

        self.system_dim = system_dim
        self.time_dim = time_dim
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
