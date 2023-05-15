"""Class for Ensemble Transform Kalman Filter (ETKF) DA Class"""

import numpy as np
from scipy import sparse

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
        assert (nr == self.ensemble_dimension,
                'cycle:: model_forecast must have dimension {}x{}'.format(
                    self.ensemble_dim,'system_dimension')
                )

        # Analysis cycles over all obs in data_obs
        Xa = self._compute_analysis(Xb=Xbt.T,
                                    y=obs_vec,
                                    H=self.H,
                                    R=self.R,
                                    rho=self.multiplicative_inflation)

        Xa = self.datoolkit.compute_analysis(Xb=Xbt.T,  # forecast ensemble (system_dimension x ensemble_dimension)
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
          
