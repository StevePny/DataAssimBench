"""Base class for Data Assimilation Cycler object (DACycler)"""

from dabench import vector
import numpy as np


class DACycler():
    """Base class for DACycler object

    Attributes:
        system_dim (int): System dimension
        delta_t (float): The timestep of the model (assumed uniform)
        model_obj (dabench.Model): Forecast model object.
        in_4d (bool): True for 4D data assimilation techniques (e.g. 4DVar).
            Default is False.
        ensemble (bool): True for ensemble-based data assimilation techniques
            (ETKF). Default is False
        B (ndarray): Initial / static background error covariance. Shape:
            (system_dim, system_dim). If not provided, will be calculated
            automatically.
        R (ndarray): Observation error covariance matrix. Shape
            (obs_dim, obs_dim). If not provided, will be calculated
            automatically.
        H (ndarray): Observation operator with shape: (obs_dim, system_dim).
            If not provided will be calculated automatically.
        h (function): Optional observation operator as function. More flexible
            (allows for more complex observation operator). Default is None.
    """

    def __init__(self,
                 system_dim=None,
                 delta_t=None,
                 model_obj=None,
                 in_4d=False,
                 ensemble=False,
                 B=None,
                 R=None,
                 H=None,
                 h=None,
                 ):

        self.h = h
        self.H = H
        self.R = R
        self.B = B
        self.in_4d = in_4d
        self.ensemble = ensemble
        self.system_dim = system_dim
        self.delta_t = delta_t
        self.model_obj = model_obj

    def cycle(self,
              input_state,
              start_time,
              obs_vector,
              n_cycles,
              analysis_window,
              analysis_time_in_window=None,
              return_forecast=False):
        """Perform DA cycle repeatedly, including analysis and forecast

        Args:
            input_state (vector.StateVector): Input state.
            start_time (float or datetime-like): Starting time.
            obs_vector (vector.ObsVector): Observations vector.
            n_cycles (int): Number of analysis cycles to run, each of length
                analysis_window.
            analysis_window (float): Time window from which to gather
                observations for DA Cycle.
            analysis_time_in_window (float): Where within analysis_window
                to perform analysis. For example, 0.0 is the start of the
                window. Default is None, which selects the middle of the
                window.
            return_forecast (bool): If True, returns forecast at each model
                timestep. If False, returns only analyses, one per analysis
                cycle. Default is False.

        Returns:
            vector.StateVector of analyses and times.
        """

        # If don't specify analysis_time_in_window, is assumed to be middle
        if analysis_time_in_window is None:
            analysis_time_in_window = analysis_window/2

        # Time offset from middle of time window, for gathering observations
        _time_offset = (analysis_window/2) - analysis_time_in_window

        # Number of model steps to run per window
        steps_per_window = round(analysis_window/self.delta_t) + 1
        print(steps_per_window)

        # For storing outputs
        all_output_states = []
        all_times = []
        cur_time = start_time
        cur_state = input_state

        for i in range(n_cycles):
            # 1. Filter observations to inside analysis window
            window_middle = cur_time + _time_offset
            window_start = window_middle - analysis_window/2
            window_end = window_middle + analysis_window/2
            obs_vec_timefilt = obs_vector.sel(
                time=slice(window_start, window_end)
            )

            if obs_vec_timefilt.sizes['time'] > 0:
                # 2. Calculate analysis
                analysis, kh = self._step_cycle(cur_state, obs_vec_timefilt)
                # 3. Forecast through analysis window
                forecast_states = self._step_forecast(analysis,
                                                      n_steps=steps_per_window)
                # 4. Save outputs
                if return_forecast:
                    # Append forecast to current state, excluding last step
                    print(forecast_states)
                    all_output_states.append(forecast_states.isel(time=slice(0,steps_per_window-1)))
                else:
                    all_output_states.append(analysis)

            # Starting point for next cycle is last step of forecast
            cur_state = forecast_states.isel(time=steps_per_window-1)
            print(cur_state)
            cur_time += analysis_window

        return all_output_states

