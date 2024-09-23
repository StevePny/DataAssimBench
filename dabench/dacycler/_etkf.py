"""Class for Ensemble Transform Kalman Filter (ETKF) DA Class"""

from functools import partial
import numpy as np
import jax
import jax.numpy as jnp
from jax.scipy import linalg
import xarray as xr
import xarray_jax as xj

from dabench import dacycler, vector
import dabench.dacycler._utils as dac_utils


class ETKF(dacycler.DACycler):
    """Class for building ETKF DA Cycler

    Attributes:
        system_dim (int): System dimension.
        ensemble_dim (int): Number of ensemble instances for ETKF. Default is
            4. Higher ensemble_dim increases accuracy but has performance cost.
        delta_t (float): The timestep of the model (assumed uniform)
        model_obj (dabench.Model): Forecast model object.
        in_4d (bool): True for 4D data assimilation techniques (e.g. 4DVar).
            Always False for ETKF.
        ensemble (bool): True for ensemble-based data assimilation techniques
            (ETKF). Always True for ETKF.
        B (ndarray): Initial / static background error covariance. Shape:
            (system_dim, system_dim). If not provided, will be calculated
            automatically.
        R (ndarray): Observation error covariance matrix. Shape
            (obs_dim, obs_dim). If not provided, will be calculated
            automatically.
        H (ndarray): Observation operator with shape: (obs_dim, system_dim).
            If not provided will be calculated automatically.
        h (function): Optional observation operator as function. More flexible
            (allows for more complex observation operator).
    """

    def __init__(self,
                 system_dim=None,
                 ensemble_dim=4,
                 delta_t=None,
                 model_obj=None,
                 multiplicative_inflation=1.0,
                 B=None,
                 R=None,
                 H=None,
                 h=None,
                 random_seed=99,
                 **kwargs
                 ):

        self.ensemble_dim = ensemble_dim
        self.random_seed = random_seed
        self._rng = np.random.default_rng(self.random_seed)
        self.multiplicative_inflation = multiplicative_inflation

        super().__init__(system_dim=system_dim,
                         delta_t=delta_t,
                         model_obj=model_obj,
                         in_4d=False,
                         ensemble=True,
                         B=B, R=R, H=H, h=h)

    def _step_cycle(self, xb, obs_vals, obs_locs, obs_time_mask, obs_loc_mask,
                    H=None, h=None, R=None, B=None):
        if H is not None or h is None:
            vals = self._cycle_obsop(
                    xb, obs_vals, obs_locs, obs_time_mask,
                    obs_loc_mask, H, R, B)
            return vals
        else:
            return self._cycle_general_obsop(xb, yo, h, R, B)

    def _calc_default_H(self, obs_values, obs_loc_indices):
        H = jnp.zeros((obs_values.flatten().shape[0], self.system_dim))
        H = H.at[jnp.arange(H.shape[0]), 
                 obs_loc_indices.flatten(),
                 ].set(1)
        return H

    def _calc_default_R(self, obs_values, obs_error_sd):
        return jnp.identity(obs_values.flatten().shape[0])*(obs_error_sd**2)

    def _calc_default_B(self):
        return jnp.identity(self.system_dim)

    def _cycle_obsop(self, x0_xarray, obs_values, obs_loc_indices,
                     obs_time_mask, obs_loc_mask,
                     H=None, h=None, R=None, B=None):
        if H is None and h is None:
            if self.H is None:
                if self.h is None:
                    H = self._calc_default_H(obs_values, obs_loc_indices)
                else:
                    h = self.h
            else:
                H = self.H
        if R is None:
            if self.R is None:
                R = self._calc_default_R(obs_values, self.obs_error_sd)
            else:
                R = self.R
        if B is None:
            if self.B is None:
                B = self._calc_default_B()
            else:
                B = self.B

        x0_xarray = x0_xarray.to_xarray()
        Xbt = x0_xarray[self._data_vars].to_array().data[0]
        nr,nc = Xbt.shape
        assert nr == self.ensemble_dim, (
                'cycle:: model_forecast must have dimension {}x{}').format(
                    self.ensemble_dim, self.system_dim)

        # Apply obs masks to H
        # H = jnp.where(obs_time_mask, H.T, 0).T
        # H = jnp.where(obs_loc_mask.flatten(), H.T, 0).T

        # Analysis cycles over all obs in data_obs
        Xa = self._compute_analysis(Xb=Xbt.T,
                                    y=obs_values,
                                    H=H,
                                    h=h,
                                    R=R,
                                    rho=self.multiplicative_inflation)

        return x0_xarray.assign(x=(['ensemble','i'], Xa.T))

    def _step_forecast(self, xa, n_steps):
        ensemble_forecasts = []
        ensemble_inputs = []
        for i in range(self.ensemble_dim):
            cur_inputs, cur_forecast = self.model_obj.forecast(
                    xa.isel(ensemble=i),
                    n_steps=n_steps
                    )
            ensemble_inputs.append(cur_inputs)
            ensemble_forecasts.append(cur_forecast)

        return (xr.concat(ensemble_inputs, dim='ensemble'),
                xr.concat(ensemble_forecasts, dim='ensemble'))

    def _apply_obsop(self, Xb, H, h):
        if H is not None:
            Yb = H @ Xb
        else:
            Yb = h(Xb)

        return Yb

    def _compute_analysis(self, Xb, y, H, h, R, rho=1.0, Yb=None):
        """ETKF analysis algorithm

        Args:
          Xb (ndarray): Forecast/background ensemble with shape
            (system_dim, ensemble_dim).
          y (ndarray): Observation array with shape (observation_dim,)
          H (ndarray): Observation operator with shape (observation_dim,
            system_dim).
          R (ndarray): Observation error covariance matrix with shape
            (observation_dim, observation_dim)
          rho (float): Multiplicative inflation factor. Default=1.0,
            (i.e. no inflation)

        Returns:
          Xa (ndarray): Analysis ensemble [size: (system_dim, ensemble_dim)]
        """
        # Number of state variables, ensemble members and observations
        system_dim, ensemble_dim = Xb.shape
        observation_dim = y.shape[0]

        # Auxiliary matrices that will ease the computations
        U = jnp.ones((ensemble_dim, ensemble_dim))/ensemble_dim
        I = jnp.identity(ensemble_dim)

        # The ensemble is inflated (rho=1.0 is no inflation)
        Xb_pert = Xb @ (I-U)
        Xb = Xb_pert + Xb @ U

        # Ensemble Transform Kalman Filter
        # Initialize the ensemble in observation space
        if Yb is None:
            Yb = jnp.empty((observation_dim, ensemble_dim))

            # Map every ensemble member into observation space
            Yb = self._apply_obsop(Xb, H, h)

        # Get ensemble means and perturbations
        xb_bar = jnp.mean(Xb,  axis=1)
        Xb_pert = Xb @ (I-U)

        yb_bar = jnp.mean(Yb, axis=1)
        Yb_pert = Yb @ (I-U)

        # Compute the analysis
        if len(R) > 0:
            Rinv = jnp.linalg.pinv(R, rtol=1e-15)

            Pa_ens = jnp.linalg.pinv((ensemble_dim-1)/rho*I
                                     + Yb_pert.T @ Rinv @ Yb_pert,
                                     rtol=1e-15)
            Wa = linalg.sqrtm((ensemble_dim-1) * Pa_ens)
            Wa = Wa.real
        else:
            Rinv = jnp.zeros_like(R, dtype=R.dtype)
            Pa_ens = jnp.zeros((ensemble_dim, ensemble_dim), dtype=R.dtype)
            Wa = jnp.zeros((ensemble_dim, ensemble_dim), dtype=R.dtype)

        wa = Pa_ens @ Yb_pert.T @ Rinv @ (y.flatten()-yb_bar)

        Xa_pert = Xb_pert @ Wa

        xa_bar = xb_bar + jnp.ravel(Xb_pert @ wa)

        v = jnp.ones((1, ensemble_dim))
        Xa = Xa_pert + xa_bar[:, None] @ v

        return Xa

    def _cycle_and_forecast(self, cur_state, filtered_idx):
        # 1. Get data
        # cur_state_vals = state[self.data_vars].data state_obs_tuple[0]

        # 1-b. Calculate obs_time_mask and restore filtered_idx to original values
        obs_time_mask = filtered_idx > 0
        filtered_idx = filtered_idx - 1

        # 2. Calculate analysis
        cur_obs_vals = jnp.array(self._obs_vector[self._observed_vars].to_array().data).at[:, filtered_idx].get()
        cur_obs_loc_indices = jnp.array(self._obs_vector.indices.data).at[filtered_idx].get()
        cur_obs_loc_mask = jnp.array(self._obs_loc_masks).at[filtered_idx].get().astype(bool)
        analysis = self._step_cycle(
                cur_state, 
                cur_obs_vals,
                cur_obs_loc_indices,
                obs_loc_mask=cur_obs_loc_mask,
                obs_time_mask=obs_time_mask
                )
        # 3. Forecast next timestep
        next_state, forecast_states = self._step_forecast(analysis, n_steps=self.steps_per_window)

        return xj.from_xarray(next_state), forecast_states

    def cycle(self,
              input_state,
              start_time,
              obs_vector,
              n_cycles,
              obs_error_sd=None,
              analysis_window=0.2,
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

        # These could be different if observer doesn't observe all variables
        # For now, making them the same
        self._observed_vars = obs_vector['variable'].values
        self._data_vars = self._observed_vars

        if obs_error_sd is None:
            obs_error_sd = obs_vector.error_sd
        self.analysis_window = analysis_window

        # If don't specify analysis_time_in_window, is assumed to be middle
        if analysis_time_in_window is None:
            analysis_time_in_window = analysis_window/2

        # Steps per window + 1 to include start
        self.steps_per_window = round(analysis_window/self.delta_t) + 1

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
        
        all_filtered_padded = dac_utils._pad_time_indices(all_filtered_idx, add_one=True)
        self._obs_vector=obs_vector
        self.obs_error_sd = obs_error_sd
        if obs_vector.stationary_observers:
            self._obs_loc_masks = jnp.ones(
                obs_vector[self._observed_vars].to_array().shape, dtype=bool)
            cur_state, all_values = jax.lax.scan(
                    self._cycle_and_forecast,
                    xj.from_xarray(input_state),
                    all_filtered_padded)
        else:
            obs_vals, obs_locs, self._obs_loc_masks = dac_utils._pad_obs_locs(obs_vector)
            cur_state, all_values = jax.lax.scan(
                    self._cycle_and_forecast,
                    (input_state.values, obs_vals, obs_vector.times,
                    obs_locs, obs_loc_masks, obs_error_sd),
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