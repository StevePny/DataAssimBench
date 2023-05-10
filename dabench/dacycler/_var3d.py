"""Class for 3D Var Data Assimilation Cycler object"""

import numpy as np
from scipy import sparse

from dabench import dacycler, vector


class Var3D(dacycler.DACycler):
    """Class for building 3DVar DA Cycler"""

    def __init__(self,
                 system_dim=None,
                 delta_t=None,
                 model_obj=None,
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
                 B=None,
                 R=None,
                 h=None,
                 H=None,
                 **kwargs
                 ):

        self.H = H
        self.R = R
        self.B = B

        super().__init__(system_dim=system_dim,
                         delta_t=delta_t,
                         model_obj=model_obj)

    def cycle(self, xb, yo, H=None, h=None, R=None, B=None):
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

        # make inputs column vectors
        xb = np.matrix(model_forecast.values).flatten().T
        yo = np.matrix(obs_vec.values).flatten().T

        # Set parameters
        xdim = xb.size  # Size or get one of the shape params?
        Rinv = np.linalg.inv(R)

        # 'preconditioning with B'
        I = np.identity(xdim)
        BHt = np.dot(B, H.T)
        BHtRinv = np.dot(BHt, Rinv)
        A = I + np.dot(BHtRinv, H)
        b1 = xb + np.dot(BHtRinv, yo)

        # Use minimization algorithm to minimize cost function:
        xa, ierr = sparse.linalg.cg(A, b1, x0=xb, tol=1e-05, maxiter=1000)

        # Compute KH:
        HBHtPlusR_inv = np.linalg.inv(H @  BHt + R)
        KH = BHt @ HBHtPlusR_inv @ H

        return vector.StateVector(values=xa), KH

    def forecast(self, xa):
        return self.model_obj.forecast(xa)
