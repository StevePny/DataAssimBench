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
              timesteps,
              analysis_window,
              analysis_time_in_window=None):
        """Perform DA cycle repeatedly, including analysis and forecast

        Args:
            input_state (vector.StateVector): Input state.
            start_time (float or datetime-like): Starting time.
            obs_vector (vector.ObsVector): Observations vector.
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

