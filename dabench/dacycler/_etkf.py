"""Class for Ensemble Transform Kalman Filter (ETKF) DA Class"""

import numpy as np
import jax.numpy as jnp
from jax.scipy import linalg

from dabench import dacycler, vector


class ETKF(dacycler.DACycler):
    """Class for building ETKF DA Cycler"""

    def __init__(self,
                 x0_pred=None,
                 system_dim=None,
                 ensemble=True,
                 ensemble_dim=4,
                 delta_t=None,
                 forecast_model=None,
                 truth_obj=None,
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
        self.R = R
        self.B = B
        self.x0_pred = x0_pred
        self.ensemble_dim = ensemble_dim
        self._rng = np.random.default_rng(random_seed)
        self.multiplicative_inflation = multiplicative_inflation

        super().__init__(system_dim=system_dim,
                         delta_t=delta_t,
                         forecast_model=forecast_model,
                         truth_obj=truth_obj,
                         ensemble_dim=ensemble_dim)

        if self.x0_pred is None:
            self.x0_pred = (self.truth_obj.x0
                            + self._rng.normal(size=(self.ensemble_dim,
                                                     self.system_dim))
                            )

    def step_cycle(self, xb, yo, H=None, h=None, R=None, B=None):
        if H is not None or h is None:
            return self._cycle_linear_obsop(xb, yo, H, R, B)
        else:
            return self._cycle_general_obsop(xb, yo, h, R, B)

    def _calc_default_H(self, obs_vec):
        H = jnp.zeros((obs_vec.values.flatten().shape[0], self.system_dim))
        H = H.at[jnp.arange(H.shape[0]), obs_vec.location_indices.flatten()
                 ].set(1)
        return H

    def _calc_default_R(self, obs_vec):
        return jnp.identity(obs_vec.values.flatten().shape[0])*obs_vec.error_sd

    def _calc_default_B(self):
        return jnp.identity(self.system_dim)

    def _cycle_general_obsop(self, model_forecast, obs_vec):
        # make inputs column vectors
        xb = model_forecast.flatten().T
        yo = obs_vec.values.flatten().T

    def _cycle_linear_obsop(self, model_forecast, obs_vec, H=None, R=None,
                            B=None):
        if H is None:
            if self.H is None:
                H = self._calc_default_H(obs_vec)
            else:
                H = self.H
        if R is None:
            if self.R is None:
                R = self._calc_default_R(obs_vec)
            else:
                R = self.R
        if B is None:
            if self.B is None:
                B = self._calc_default_B()
            else:
                B = self.B

        Xbt = model_forecast.values
        nr, nc = Xbt.shape
        assert nr == self.ensemble_dim, (
                'cycle:: model_forecast must have dimension {}x{}').format(
                    self.ensemble_dim, self.system_dim)
                

        # Analysis cycles over all obs in data_obs
        Xa = self._compute_analysis(Xb=Xbt.T,
                                    y=obs_vec,
                                    H=H,
                                    R=R,
                                    rho=self.multiplicative_inflation)

        Xat = Xa.T

        return vector.StateVector(values=Xat, store_as_jax=True), 0


    def step_forecast(self, xa):
        data_forecast = []
        for i in range(self.ensemble_dim):
            new_vals = self.forecast_model.forecast(
                    vector.StateVector(values=xa.values[i], store_as_jax=True)
                    ).values
            data_forecast.append(new_vals)

        return vector.StateVector(values=jnp.stack(data_forecast), store_as_jax=True)

    def _compute_analysis(self, Xb, y, H, R, rho=1.0, Yb=None):
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

        y = y.values

        # Number of state variables, ensemble members and observations
        system_dim, ensemble_dim = Xb.shape
        observation_dim, system_dim = H.shape

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
            try:
                Yb = H @ Xb
            except:
                print('Yb.shape = {}, H.shape = {}, Xb.shape = {}'.format(Yb.shape,H.shape,Xb.shape))
                raise

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
            Wa = np.real_if_close(Wa)
        else:
            Rinv = jnp.zeros_like(R,dtype=R.dtype)
            Pa_ens = jnp.zeros((ensemble_dim, ensemble_dim), dtype=R.dtype)
            Wa = jnp.zeros((ensemble_dim, ensemble_dim), dtype=R.dtype)

        try:
            wa = Pa_ens @ Yb_pert.T @ Rinv @ (y.flatten()-yb_bar)
        except:
            print('Pa_ens.shape = {}, Yb_pert.shape = {} Rinv.shape = {}, y.shape = {}, yb_bar.shape = {}'.format(Pa_ens.shape, Yb_pert.shape, Rinv.shape, y.shape, yb_bar.shape))
            print('If y.shape is incorrect, make sure that the S operator is defined correctly at input.')
            raise

        Xa_pert = Xb_pert @ Wa

        try:
            xa_bar = xb_bar + jnp.ravel(Xb_pert @ wa)
        except:
            print('xb_bar.shape = {}, Xb_pert.shape = {} wa.shape = {}'.format(xb_bar.shape,Xb_pert.shape,wa.shape))
            raise

        v = jnp.ones((1, ensemble_dim))
        try:
            Xa = Xa_pert + xa_bar[:, None] @ v
        except:
            print('Xa_pert.shape = {}, xa_bar.shape = {} v.shape = {}'.format(Xa_pert.shape,xa_bar.shape,v.shape))
            print('xb_bar.shape = {}, Xb_pert.shape = {} wa.shape = {}'.format(xb_bar.shape,Xb_pert.shape,wa.shape))
            raise

        return Xa

