"""Base class for Data Assimilation Cycler object (DACycler)"""


class DACycler():
    """Base class for DACycler object

    Attributes:
        system_dim (int): system dimension
        time_dim (int): total time steps
        delta_t (float): the timestep of the model (assumed uniform)
        model_obj (obj): underlying model object, e.g. pytorch neural network.
    """
    def __init__(self,
                 start_time=0,
                 end_time=None,
                 num_cycles=1,
                 window_time=None,
                 in_4d=False,
                 ensemble=False,
                 analysis_window=None,
                 observation_window=None,
                 observations=None,
                 forecast_model=None,
                 B_matrix=None,
                 R_matrix=None,
                 model_obj=None,
                 **kwargs
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

    def _default_forecast(self, state_vec, timesteps=1, other_inputs=None):
        """Default method for forecasting"""
        new_state_vec = state_vec
        for i in range(timesteps):
            new_state_vec = self.model_obj.predict(new_state_vec)

        return new_state_vec
