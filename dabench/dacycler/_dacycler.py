"""Base class for Data Assimilation Cycler object (DACycler)"""

import numpy as np
import jax.numpy as jnp
import jax
import xarray as xr
import xarray_jax as xj

import dabench.dacycler._utils as dac_utils

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


    def _step_forecast(self, xa, n_steps=1):
        """Perform forecast using model object"""
        return self.model_obj.forecast(xa, n_steps=n_steps)

    def _step_cycle(self, xb, obs_vals, obs_locs, obs_time_mask, obs_loc_mask,
                    H=None, h=None, R=None, B=None, **kwargs):
        if H is not None or h is None:
            vals = self._cycle_obsop(
                    xb, obs_vals, obs_locs, obs_time_mask,
                    obs_loc_mask, H, R, B, **kwargs)
            return vals
        else:
            return self._cycle_general_obsop(xb, yo, h, R, B)

    def cycle(self,
              input_state,
              start_time,
              obs_vector,
              n_cycles,
              obs_error_sd=None,
              analysis_window=0.2,
              analysis_time_in_window=None,
              return_forecast=False
              ):
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

        # These could be different if observer doesn't observe all variables
        # For now, making them the same
        self._observed_vars = obs_vector['variable'].values
        self._data_vars = list(input_state.data_vars)

        if obs_error_sd is None:
            obs_error_sd = obs_vector.error_sd
        self.analysis_window = analysis_window

        # If don't specify analysis_time_in_window, is assumed to be middle
        if analysis_time_in_window is None:
            analysis_time_in_window = analysis_window/2

        # Steps per window + 1 to include start
        self.steps_per_window = round(analysis_window/self.delta_t) + 1
        self._model_timesteps = jnp.arange(self.steps_per_window)*self.delta_t

        # Time offset from middle of time window, for gathering observations
        _time_offset = (analysis_window/2) - analysis_time_in_window

        # Set up for jax.lax.scan, which is very fast
        all_times = dac_utils._get_all_times(
            start_time,
            analysis_window,
            n_cycles)
            
        # Get the obs vectors for each analysis window
        all_filtered_idx = dac_utils._get_obs_indices(
            obs_times=jnp.array(obs_vector.time.values),
            analysis_times=all_times+_time_offset,
            start_inclusive=True,
            end_inclusive=False,
            analysis_window=analysis_window
        )
        input_state = input_state.assign(_cur_time=start_time)
        
        all_filtered_padded = dac_utils._pad_time_indices(all_filtered_idx, add_one=True)
        self._obs_vector=obs_vector
        self.obs_error_sd = obs_error_sd
        if obs_vector.stationary_observers:
            self._obs_loc_masks = jnp.ones(
                obs_vector[self._observed_vars].to_array().shape, dtype=bool)
        else:
            self._obs_loc_masks = ~np.isnan(
                obs_vector[self._observed_vars].to_array().data)[0]
            self._obs_vector=self._obs_vector.fillna(0)
        cur_state, all_values = jax.lax.scan(
                self._cycle_and_forecast,
                xj.from_xarray(input_state),
                all_filtered_padded)
                
        all_vals_xr = xr.Dataset(
            {var: (('cycle',) + tuple(all_values[var].dims),
                   all_values[var].data)
             for var in all_values.data_vars}
        ).rename_dims({'time': 'cycle_timestep'})

        if return_forecast:
            return all_vals_xr
        else:
            return all_vals_xr.isel(cycle_timestep=0)