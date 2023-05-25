"""Base class for Data Assimilation Cycler object (DACycler)"""

from dabench import vector
import numpy as np


class DACycler():
    """Base class for DACycler object

    Attributes:
        system_dim (int): system dimension
        delta_t (float): the timestep of the model (assumed uniform)
        forecast_model (obj): underlying model object, e.g.
            pytorch neural network.
        in_4d (bool): True for 4D data assimilation techniques (e.g. 4DVar).
            Default is False.
        ensemble (bool): True for ensemble-based data assimilation techniques (ETKF).
            Default is False
    """
    def __init__(self,
                 system_dim=None,
                 delta_t=None,
                 forecast_model=None,
                 in_4d=False,
                 ensemble=False,
                 **kwargs
                 ):

        self.in_4d = in_4d
        self.ensemble = ensemble
        self.system_dim = system_dim
        self.delta_t = delta_t
        self.forecast_model = forecast_model

    def cycle(self,
              input_state,
              start_time,
              obs_vector,
              timesteps,
              analysis_window,
              analysis_time_in_window=None):
        """Perform DA cycle repeatedly, including analysis and forecast

        Args:
            input_state (vector.StateVector): Input state.
            obs_vector (vector.ObsVector): Observations vector.
            start_time (float or datetime-like): Starting time.
            timesteps (int): Number of timesteps, in model time.
            analysis_window (float): Time window from which to gather
                observations for DA Cycle.
            analysis_time_in_window (float): Where within analysis_window
                to perform analysis. For example, 0.0 is the start of the
                window. Default is None, which selects the middle of the
                window.

        Returns:
            vector.StateVector of analyses and times.
        """

        if analysis_time_in_window is None:
            analysis_time_in_window = analysis_window/2
        # For storing outputs
        all_analyses = []
        all_times = []
        cur_time = start_time + analysis_time_in_window
        cur_state = input_state

        for i in range(timesteps):
            # 1. Filter observations to plus/minus 0.1 from that time
            obs_vec_timefilt = obs_vector.filter_times(
                cur_time - analysis_window/2, cur_time + analysis_window/2)

            if obs_vec_timefilt.values.shape[0] > 0:
                # 2. Calculate analysis
                analysis, kh = self.step_cycle(cur_state, obs_vec_timefilt)
                # 3. Forecast next timestep
                cur_state = self.step_forecast(analysis)
                # 4. Save outputs
                all_analyses.append(analysis.values)
                all_times.append(cur_time)

            cur_time += self.delta_t

        return vector.StateVector(values=np.stack(all_analyses),
                                  times=np.array(all_times))


