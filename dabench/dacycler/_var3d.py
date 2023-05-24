"""Class for 3D Var Data Assimilation Cycler object"""

import numpy as np
import jax.numpy as jnp
import jax.scipy as jscipy

from dabench import dacycler, vector


class Var3D(dacycler.DACycler):
    """Class for building 3DVar DA Cycler"""

    def __init__(self,
                 system_dim=None,
                 delta_t=None,
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
                         forecast_model=forecast_model)

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

    def _cycle_general_obsop(self, forecast, obs_vec):
        # make inputs column vectors
        xb = forecast.flatten().T
        yo = obs_vec.values.flatten().T

    def _cycle_linear_obsop(self, forecast, obs_vec, H=None, R=None,
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
        xb = jnp.array([forecast.values.flatten()]).T
        yo = jnp.array([obs_vec.values.flatten()]).T

        # Set parameters
        xdim = xb.size  # Size or get one of the shape params?
        Rinv = jnp.linalg.inv(R)

        # 'preconditioning with B'
        I = jnp.identity(xdim)
        BHt = jnp.dot(B, H.T)
        BHtRinv = jnp.dot(BHt, Rinv)
        A = I + jnp.dot(BHtRinv, H)
        b1 = xb + jnp.dot(BHtRinv, yo)

        # Use minimization algorithm to minimize cost function:
        xa, ierr = jscipy.sparse.linalg.cg(A, b1, x0=xb, tol=1e-05,
                                           maxiter=1000)

        # Compute KH:
        HBHtPlusR_inv = jnp.linalg.inv(H @  BHt + R)
        KH = BHt @ HBHtPlusR_inv @ H

        return vector.StateVector(values=xa, store_as_jax=True), KH

    def step_forecast(self, xa):
        return self.forecast_model.forecast(xa)
