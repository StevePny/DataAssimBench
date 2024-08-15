"""Class for 3D Var Data Assimilation Cycler object"""

import numpy as np
import jax.numpy as jnp
import jax.scipy as jscipy

from dabench import dacycler, vector


class Var3D(dacycler.DACycler):
    """Class for building 3DVar DA Cycler

    Attributes:
        system_dim (int): System dimension.
        delta_t (float): The timestep of the model (assumed uniform)
        model_obj (dabench.Model): Forecast model object.
        in_4d (bool): True for 4D data assimilation techniques (e.g. 4DVar).
            Always False for Var3D.
        ensemble (bool): True for ensemble-based data assimilation techniques
            (ETKF). Always False for Var3D
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
                 in_4d=False,
                 model_obj=None,
                 B=None,
                 R=None,
                 H=None,
                 h=None,
                 ):

        super().__init__(system_dim=system_dim,
                         delta_t=delta_t,
                         model_obj=model_obj,
                         in_4d=False,
                         ensemble=False,
                         B=B, R=R, H=H, h=h)

    def _step_cycle(self, xb, yo, H=None, h=None, R=None, B=None):
        """Perform one step of DA Cycle

        Args:
            xb: 
            yo:
            H


        Returns:
            vector.StateVector containing analysis results

        """
        if H is not None or h is None:
            return self._cycle_linear_obsop(xb, yo, H, R, B)
        else:
            return self._cycle_general_obsop(xb, yo, h, R, B)

    def _calc_default_H(self, obs_vec):
        """If H is not provided, creates identity matrix to serve as H"""
        H = jnp.zeros((obs_vec.values.flatten().shape[0], self.system_dim))
        H = H.at[jnp.arange(H.shape[0]), obs_vec.location_indices.flatten()
                 ].set(1)
        return H

    def _calc_default_R(self, obs_vec):
        """If R i s not provided, calculates default based on observation error"""
        return jnp.identity(obs_vec.values.flatten().shape[0])*obs_vec.error_sd**2

    def _calc_default_B(self):
        """If B is not provided, identity matrix with shape (system_dim, system_dim."""

        return jnp.identity(self.system_dim)

    def _cycle_general_obsop(self, forecast, obs_vec):
        return

    def _cycle_linear_obsop(self, forecast, obs_vec, H=None, R=None,
                            B=None):
        """When obsop (H) is linear"""
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

        return vector.StateVector(values=xa.T[0], store_as_jax=True), KH

    def _step_forecast(self, xa, n_steps=1):
        """n_steps forward of model forecast"""
        return self.model_obj.forecast(xa, n_steps=n_steps)
