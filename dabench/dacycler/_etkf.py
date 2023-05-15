"""Class for Ensemble Transform Kalman Filter (ETKF) DA Class"""

import numpy as np
from scipy import sparse, linalg

from dabench import dacycler, vector


class ETKF(dacycler.DACycler):
    """Class for building ETKF DA Cycler"""

    def __init__(self,
                 x0_pred=None,
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
                 forecast_model=None,
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
        if self.x0_pred is None:
            self.x0_pred = (self.model_obj.x0
                            + self._rng.normal(self.ensemble_dim,
                                               self.system_dim)
                            )

        super().__init__(system_dim=system_dim,
                         delta_t=delta_t,
                         model_obj=model_obj,
                         ensemble_dim=ensemble_dim)

    def step_cycle(self, xb, yo, H=None, h=None, R=None, B=None):
        if H is not None or h is None:
            return self._cycle_linear_obsop(xb, yo, H, R, B)
        else:
            return self._cycle_general_obsop(xb, yo, h, R, B)

    def _calc_default_H(self, obs_vec):
        H = np.zeros((obs_vec.values.flatten().shape[0], self.system_dim))
        H[np.arange(H.shape[0]), obs_vec.location_indices.flatten()] = 1
        return H

    def _calc_default_R(self, obs_vec):
        return np.identity(obs_vec.values.flatten().shape[0])*obs_vec.error_sd

    def _calc_default_B(self):
        return np.identity(self.system_dim)

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

        Xbt = model_forecast
        nr, nc = Xbt.shape
        assert (nr == self.ensemble_dim,
                'cycle:: model_forecast must have dimension {}x{}'.format(
                    self.ensemble_dim, self.system_dim)
                )

        # Analysis cycles over all obs in data_obs
        Xa = self._compute_analysis(Xb=Xbt.T,
                                    y=obs_vec,
                                    H=self.H,
                                    R=self.R,
                                    rho=self.multiplicative_inflation)

        Xa = self.datoolkit.compute_analysis(Xb=Xbt.T,  # forecast ensemble (system_dim x ensemble_dim)
                                             y=self.data_obs.values[i],   # observations for this time
                                             H=self.H,                            # observation operator
                                             R=self.R,                            # observation error covariance
                                             rho=self.multiplicative_inflation)   # multiplicative inflation parameter
        Xat = Xa.T

        # Generate an ensemble forecast
        # Output is [ens, time, system]
        ens_fcst = self.step_forecast(np.array(Xat)) 

        # Take the next cycle's background as the last timestep of the forecast
        Xbt = ens_fcst[:, -1, :]

    def step_forecast(self, xa):
        data_forecast = []
        for i in range(self.ensemble_dim):
            self.model_obj.forecast(x0=xa[i])
            data_forecast.append(self.model_obj.values)

        return np.stack(data_forecast)
          

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

        # Input checks
        assert isinstance(Xb,np.ndarray), 'observation vector Xb must be an ndarray, instead type(Xb) = {}'.format(type(Xb))
        assert isinstance(y,np.ndarray), 'observation vector y must be an ndarray, instead type(y) = {}'.format(type(y))
        assert isinstance(H,np.ndarray), 'observation vector H must be an ndarray, instead type(H) = {}'.format(type(H))
        assert isinstance(R,np.ndarray), 'observation vector R must be an ndarray, instead type(R) = {}'.format(type(R))

        # Number of state variables, ensemble members and observations
        system_dim, ensemble_dim    = Xb.shape
        observation_dim, system_dim = H.shape

        # Auxiliary matrices that will ease the computation of averages and covariances
        U = np.ones((ensemble_dim, ensemble_dim))/ensemble_dim
        I = np.identity(ensemble_dim)

        # The ensemble is inflated (rho=1.0 is no inflation)
        #ISSUE: this can be applied with a single multiply below - see Hunt et al. (2007)
        Xb_pert = Xb @ (I-U)
        Xb = Xb_pert + Xb @ U

        # Ensemble Transform Kalman Filter
        # Initialize the ensemble in observation space
        if Yb is None:
            Yb = np.mat(np.empty((observation_dim, ensemble_dim)))
            Yb.fill(np.nan)

            # Map every ensemble member into observation space
            try:
                Yb = H @ Xb
            except:
                print('Yb.shape = {}, H.shape = {}, Xb.shape = {}'.format(Yb.shape,H.shape,Xb.shape))
                raise

        # Get ensemble means and perturbations
        xb_bar = np.mean(Xb,  axis=1)
        Xb_pert = Xb @ (I-U)

        yb_bar = np.mean(Yb, axis=1)
        Yb_pert = Yb @ (I-U)

        # Compute the analysis
        # Only do this part if we have observations on this chunk (parallel case)
        if len(R) > 0:
            Rinv = linalg.pinv(R)

            Pa_ens = linalg.pinv((ensemble_dim-1)/rho*I + Yb_pert.T @ Rinv @ Yb_pert)
            Wa = linalg.sqrtm((ensemble_dim-1) * Pa_ens)  # matrix square root (symmetric)
            Wa = np.real_if_close(Wa)
        else:
            Rinv = np.zeros_like(R,dtype=R.dtype)
            Pa_ens = np.zeros((ensemble_dim, ensemble_dim), dtype=R.dtype)
            Wa = np.zeros((ensemble_dim, ensemble_dim), dtype=R.dtype)

        try:
            wa = Pa_ens @ Yb_pert.T @ Rinv @ (y-yb_bar)
        except:
            print('Pa_ens.shape = {}, Yb_pert.shape = {} Rinv.shape = {}, y.shape = {}, yb_bar.shape = {}'.format(Pa_ens.shape, Yb_pert.shape, Rinv.shape, y.shape, yb_bar.shape))
            print('If y.shape is incorrect, make sure that the S operator is defined correctly at input.')
            raise

        Xa_pert = Xb_pert @ Wa

        try:
            xa_bar = xb_bar + np.ravel(Xb_pert @ wa)
        except:
            print('xb_bar.shape = {}, Xb_pert.shape = {} wa.shape = {}'.format(xb_bar.shape,Xb_pert.shape,wa.shape))
            raise

        v = np.ones((1, ensemble_dim))
        try:
            Xa = Xa_pert + xa_bar[:, None] @ v
        except:
            print('Xa_pert.shape = {}, xa_bar.shape = {} v.shape = {}'.format(Xa_pert.shape,xa_bar.shape,v.shape))
            print('xb_bar.shape = {}, Xb_pert.shape = {} wa.shape = {}'.format(xb_bar.shape,Xb_pert.shape,wa.shape))
            raise

        return Xa

