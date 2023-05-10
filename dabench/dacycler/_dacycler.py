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
                 system_dim=None,
                 delta_t=None,
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
        self.delta_t = delta_t
        self.model_obj = model_obj
