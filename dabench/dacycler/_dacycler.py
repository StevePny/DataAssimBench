"""Base class for Data Assimilation Cycler object (DACycler)"""

from dabench import vector
import numpy as np


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

        self.ensemble = ensemble
        self.system_dim = system_dim
        self.delta_t = delta_t
        self.model_obj = model_obj

    def cycle(self,
              input_state,
              start_time,
              obs_vector,
              timesteps,
              obs_time_window=0.2):
        """Perform DA cycle repeatedly, including analysis and forecast

        Args:
            input_state (vector.StateVector): Input state.
            obs_vector (vector.ObsVector): Observations vector.
            start_time (float or datetime-like): Starting time.
            timesteps (int): Number of timesteps, in model time.
            obs_time_window (float): Time window from which to gather
                observations for DA Cycle. Takes observations that are +/-
                obs_time_window from time of each analysis step.

        Returns:
            vector.StateVector of analyses and times.
        """
        # For storing outputs
        all_analyses = []
        all_times = []
        cur_time = start_time
        cur_state = input_state

        for i in range(timesteps):
            # 1. Filter observations to plus/minus 0.1 from that time
            obs_vec_timefilt = obs_vector.filter_times(
                cur_time - obs_time_window, cur_time + obs_time_window)

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


