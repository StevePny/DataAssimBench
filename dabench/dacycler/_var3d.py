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
                 in_4d=False,
                 ensemble=False,
                 forecast_model=None,
                 B=None,
                 R=None,
                 h=None,
                 H=None,
                 **kwargs
                 ):

        self.h = h
        self.H = H
        self.R = R
        self.B = B

        super().__init__(system_dim=system_dim,
                         delta_t=delta_t,
                         forecast_model=forecast_model)

    def step_cycle(self, x_b, y_o, H=None, h=None, R=None, B=None):
        """Perform one step of DA Cycle

        Args:
            x_b:
            y_o:
            H


        Returns:
            vector.StateVector containing analysis results

        """
        if H is not None or h is None:
            return self._cycle_linear_obsop(x_b, y_o, H, R, B)
        else:
            return self._cycle_general_obsop(x_b, y_o, h, R, B)

    def _calc_default_H(self, obs_vec):
        """If H is not provided, creates identity matrix to serve as H"""
        H = jnp.zeros((obs_vec.values.flatten().shape[0], self.system_dim))
        H = H.at[jnp.arange(H.shape[0]), obs_vec.location_indices.flatten()
                 ].set(1)
        return H

    def _calc_default_R(self, obs_vec):
        """If R i s not provided, calculates default based on observation error"""
        return jnp.identity(obs_vec.values.flatten().shape[0])*obs_vec.error_sd^2

    def _calc_default_B(self):
        """If B is not provided, identity matrix with shape (system_dim, system_dim."""

        return jnp.identity(self.system_dim)

    def _cycle_general_obsop(self, forecast, obs_vec):
        return

    def _cycle_linear_obsop(self, forecast, obs_vec, H=None, R=None,
                            B=None):
        """When obsop (H/h) is linear"""
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
        x_b = jnp.array([forecast.values.flatten()]).T
        y_o = jnp.array([obs_vec.values.flatten()]).T

        # Set parameters
        xdim = x_b.size  # Size or get one of the shape params?
        Rinv = jnp.linalg.inv(R)

        # 'preconditioning with B'
        I = jnp.identity(xdim)
        BHt = jnp.dot(B, H.T)
        BHtRinv = jnp.dot(BHt, Rinv)
        A = I + jnp.dot(BHtRinv, H)
        b1 = x_b + jnp.dot(BHtRinv, y_o)

        # Use minimization algorithm to minimize cost function:
        xa, ierr = jscipy.sparse.linalg.cg(A, b1, x0=x_b, tol=1e-05,
                                           maxiter=1000)

        # Compute KH:
        HBHtPlusR_inv = jnp.linalg.inv(H @  BHt + R)
        KH = BHt @ HBHtPlusR_inv @ H

        return vector.StateVector(values=xa, store_as_jax=True), KH

    def step_forecast(self, xa):
        """One step of the forecast."""
        return self.forecast_model.forecast(xa)
