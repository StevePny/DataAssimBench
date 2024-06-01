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

    def _step_cycle(self, xb, yo, H=None, h=None, R=None, B=None):
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

    def _step_forecast(self, xa):
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
            Rinv = jnp.linalg.pinv(R, rcond=1e-15)

            Pa_ens = jnp.linalg.pinv((ensemble_dim-1)/rho*I
                                     + Yb_pert.T @ Rinv @ Yb_pert,
                                     rcond=1e-15)
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

    def _cycle_and_forecast(self, state_obs_tuple, filtered_idx):
        cur_state_vals = state_obs_tuple[0]
        obs_vals = state_obs_tuple[1]
        obs_times = state_obs_tuple[2]
        obs_loc_indices = state_obs_tuple[3]
        obs_error_sd = state_obs_tuple[4]

        # 2. Calculate analysis
        new_obs_vals = jax.lax.dynamic_slice_in_dim(obs_vals, filtered_idx[0],
                                                    len(filtered_idx))
        new_obs_loc_indices = jax.lax.dynamic_slice_in_dim(
                obs_loc_indices, filtered_idx[0], len(filtered_idx))
        analysis, kh = self._step_cycle(
                vector.StateVector(values=cur_state_vals, store_as_jax=True),
                vector.ObsVector(values=new_obs_vals,
                                 location_indices=new_obs_loc_indices,
                                 error_sd=obs_error_sd, store_as_jax=True)
                )
        # 3. Forecast next timestep
        cur_state = self._step_forecast(analysis)

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

        if obs_error_sd is None:
            obs_error_sd = obs_vector.error_sd
        self.analysis_window = analysis_window
        all_times = (jnp.repeat(start_time, timesteps)
                     + (jnp.arange(0, timesteps)*self.delta_t))
        # Get the obs vectors for each analysis window
        all_filtered_idx = jnp.stack([jnp.where(
            # Greater than start of window
            (obs_vector.times > cur_time - analysis_window/2)
            # AND Less than end of window
            * (obs_vector.times < cur_time + analysis_window/2)
            # OR Equal to start of window
            + jnp.isclose(obs_vector.times, cur_time - analysis_window/2,
                          rtol=0)
            )[0] for cur_time in all_times])
        cur_state, all_values = jax.lax.scan(
                self._cycle_and_forecast,
                (input_state.values, obs_vector.values, obs_vector.times,
                 obs_vector.location_indices, obs_error_sd),
                all_filtered_idx)

        return vector.StateVector(values=jnp.stack(all_values),
                                  times=all_times)
