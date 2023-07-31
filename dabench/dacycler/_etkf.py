"""Class for Ensemble Transform Kalman Filter (ETKF) DA Class"""

from functools import partial
import numpy as np
import jax
import jax.numpy as jnp
from jax.scipy import linalg

from dabench import dacycler, vector


class ETKF(dacycler.DACycler):
    """Class for building ETKF DA Cycler


    Attributes:



    """

    def __init__(self,
                 system_dim=None,
                 ensemble=True,
                 ensemble_dim=4,
                 delta_t=None,
                 model_obj=None,
                 start_time=0,
                 end_time=None,
                 num_cycles=1,
                 window_time=None,
                 in_4d=False,
                 analysis_window=None,
                 observation_window=None,
                 observations=None,
                 multiplicative_inflation=1.0,
                 B=None,
                 R=None,
                 h=None,
                 H=None,
                 random_seed=99,
                 **kwargs
                 ):

        self.H = H
        self.h = h
        self.R = R
        self.B = B
        self.ensemble_dim = ensemble_dim
        self._rng = np.random.default_rng(random_seed)
        self.multiplicative_inflation = multiplicative_inflation

        super().__init__(system_dim=system_dim,
                         delta_t=delta_t,
                         model_obj=model_obj,
                         ensemble_dim=ensemble_dim)

    def step_cycle(self, xb, yo, H=None, h=None, R=None, B=None):
        if H is not None or h is None:
            vals, kh = self._cycle_obsop(
                    xb.values, yo.values, yo.location_indices, yo.error_sd,
                    H, R, B)
            return vector.StateVector(values=vals, store_as_jax=True), kh
        else:
            return self._cycle_general_obsop(xb, yo, h, R, B)

    def _calc_default_H(self, obs_values, obs_loc_indices):
        H = jnp.zeros((obs_values.flatten().shape[0], self.system_dim))
        H = H.at[jnp.arange(H.shape[0]), obs_loc_indices.flatten()
                 ].set(1)
        return H

    def _calc_default_R(self, obs_values, obs_error_sd):
        return jnp.identity(obs_values.flatten().shape[0])*(obs_error_sd**2)

    def _calc_default_B(self):
        return jnp.identity(self.system_dim)

    def _cycle_obsop(self, Xbt, obs_values, obs_loc_indices, obs_error_sd,
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
                R = self._calc_default_R(obs_values, obs_error_sd)
            else:
                R = self.R
        if B is None:
            if self.B is None:
                B = self._calc_default_B()
            else:
                B = self.B

        nr, nc = Xbt.shape
        assert nr == self.ensemble_dim, (
                'cycle:: model_forecast must have dimension {}x{}').format(
                    self.ensemble_dim, self.system_dim)

        # Analysis cycles over all obs in data_obs
        Xa = self._compute_analysis(Xb=Xbt.T,
                                    y=obs_values,
                                    H=H,
                                    h=h,
                                    R=R,
                                    rho=self.multiplicative_inflation)

        return Xa.T, 0

    def step_forecast(self, xa):
        data_forecast = []
        for i in range(self.ensemble_dim):
            new_vals = self.model_obj.forecast(
                    vector.StateVector(values=xa.values[i], store_as_jax=True)
                    ).values
            data_forecast.append(new_vals)

        return vector.StateVector(values=jnp.stack(data_forecast),
                                  store_as_jax=True)

    def _apply_obsop(self, Xb, H, h):
        if H is not None:
            try:
                Yb = H @ Xb
            except TypeError:
                print('Yb.shape = {}, H.shape = {}, Xb.shape = {}'.format(
                    Yb.shape, H.shape, Xb.shape))
                raise
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

        # Auxiliary matrices that will ease the computation of averages and covariances
        U = jnp.ones((ensemble_dim, ensemble_dim))/ensemble_dim
        I = jnp.identity(ensemble_dim)

        # The ensemble is inflated (rho=1.0 is no inflation)
        #ISSUE: this can be applied with a single multiply below - see Hunt et al. (2007)
        Xb_pert = Xb @ (I-U)
        Xb = Xb_pert + Xb @ U

        # Ensemble Transform Kalman Filter
        # Initialize the ensemble in observation space
        if Yb is None:
            Yb = jnp.empty((observation_dim, ensemble_dim))
            # Yb.fill(jnp.nan) # Commenting out on 5/24, don't think this is needed

            # Map every ensemble member into observation space
            Yb = self._apply_obsop(Xb, H, h)

        # Get ensemble means and perturbations
        xb_bar = jnp.mean(Xb,  axis=1)
        Xb_pert = Xb @ (I-U)

        yb_bar = jnp.mean(Yb, axis=1)
        Yb_pert = Yb @ (I-U)

        # Compute the analysis
        # Only do this part if we have observations on this chunk (parallel case)
        if len(R) > 0:
            Rinv = jnp.linalg.pinv(R, rcond=1e-15)

            Pa_ens = jnp.linalg.pinv((ensemble_dim-1)/rho*I + Yb_pert.T @ Rinv @ Yb_pert,
                                     rcond=1e-15)
            Wa = linalg.sqrtm((ensemble_dim-1) * Pa_ens)  # matrix square root (symmetric)
            Wa = Wa.real
#             if Wa.imag.max() > jnp.finfo(float).eps*100:
#                 raise ValueError('Wa has a non-neglible imaginary component, max of {}'.format(Wa.imag.max()))
        else:
            Rinv = jnp.zeros_like(R,dtype=R.dtype)
            Pa_ens = jnp.zeros((ensemble_dim, ensemble_dim), dtype=R.dtype)
            Wa = jnp.zeros((ensemble_dim, ensemble_dim), dtype=R.dtype)

        try:
            wa = Pa_ens @ Yb_pert.T @ Rinv @ (y.flatten()-yb_bar)
        except TypeError:
            print('Pa_ens.shape = {}, Yb_pert.shape = {} Rinv.shape = {}, y.shape = {}, yb_bar.shape = {}'.format(Pa_ens.shape, Yb_pert.shape, Rinv.shape, y.shape, yb_bar.shape))
            print('If y.shape is incorrect, make sure that the S operator is defined correctly at input.')
            raise

        Xa_pert = Xb_pert @ Wa

        try:
            xa_bar = xb_bar + jnp.ravel(Xb_pert @ wa)
        except TypeError:
            print('xb_bar.shape = {}, Xb_pert.shape = {} wa.shape = {}'.format(xb_bar.shape,Xb_pert.shape,wa.shape))
            raise

        v = jnp.ones((1, ensemble_dim))
        try:
            Xa = Xa_pert + xa_bar[:, None] @ v
        except TypeError:
            print('Xa_pert.shape = {}, xa_bar.shape = {} v.shape = {}'.format(Xa_pert.shape,xa_bar.shape,v.shape))
            print('xb_bar.shape = {}, Xb_pert.shape = {} wa.shape = {}'.format(xb_bar.shape,Xb_pert.shape,wa.shape))
            raise

        return Xa

    def _cycle_and_forecast(self, state_obs_tuple, filtered_idx):
        cur_state_vals = state_obs_tuple[0]
        obs_vals = state_obs_tuple[1]
        obs_times = state_obs_tuple[2]
        obs_loc_indices = state_obs_tuple[3]
        obs_error_sd = state_obs_tuple[4]

        # 2. Calculate analysis
        new_obs_vals = jax.lax.dynamic_slice_in_dim(obs_vals, filtered_idx[0], len(filtered_idx))
        new_obs_loc_indices = jax.lax.dynamic_slice_in_dim(obs_loc_indices, filtered_idx[0], len(filtered_idx))
        analysis, kh = self.step_cycle(vector.StateVector(values=cur_state_vals, store_as_jax=True),
                                       vector.ObsVector(values=new_obs_vals, location_indices=new_obs_loc_indices, error_sd=obs_error_sd, store_as_jax=True))
        # 3. Forecast next timestep
        cur_state = self.step_forecast(analysis)

        return (cur_state.values, obs_vals, obs_times, obs_loc_indices,
                obs_error_sd), cur_state.values

    def cycle(self,
              input_state,
              start_time,
              obs_vector,
              timesteps,
              obs_error_sd=None,
              analysis_window=0.2):
        """Perform DA cycle repeatedly, including analysis and forecast

        Args:
            input_state (vector.StateVector): Input state.
            obs_vector (vector.ObsVector): Observations vector.
            obs_error_sd (float): Standard deviation of observation error.
                Typically not known, so provide a best-guess.
            start_time (float or datetime-like): Starting time.
            timesteps (int): Number of timesteps, in model time.
            analysi_window (float): Time window from which to gather
                observations for DA Cycle. Takes observations that are +/-
                obs_time_window from time of each analysis step.

        Returns:
            vector.StateVector of analyses and times.
        """
        if obs_error_sd is None:
            obs_error_sd = obs_vector.error_sd
        self.analysis_window = analysis_window
        all_times = (jnp.repeat(start_time, timesteps)
                     + jnp.arange(0, timesteps*self.delta_t, self.delta_t))
        all_filtered_idx = jnp.stack([jnp.where(
            (obs_vector.times
             >= jnp.round(cur_time - self.analysis_window/2, 3))
            * (obs_vector.times
               < jnp.round(cur_time + self.analysis_window/2, 3)))[0]
            for cur_time in all_times])
        cur_state, all_values = jax.lax.scan(
                self._cycle_and_forecast,
                (input_state.values, obs_vector.values, obs_vector.times,
                 obs_vector.location_indices, obs_error_sd),
                all_filtered_idx)

        return vector.StateVector(values=jnp.stack(all_values),
                                  times=all_times)
