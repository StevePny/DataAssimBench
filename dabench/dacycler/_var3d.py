"""Class for 3D Var Data Assimilation Cycler object"""

import numpy as np
import jax.numpy as jnp
import jax.scipy as jscipy

from dabench import dacycler


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

    def _cycle_obsop(self, x0_xarray, obs_values, obs_loc_indices,
                     obs_time_mask, obs_loc_mask,
                     H=None, h=None, R=None, B=None):
        """When obsop (H) is linear"""
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

        xb = x0_xarray.to_stacked_array('system',[]).data.flatten()
        yo = obs_values.flatten()

        # Apply masks to H
        H = jnp.where(obs_time_mask.flatten(), H.T, 0).T
        H = jnp.where(obs_loc_mask.flatten(), H.T, 0).T

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

        return x0_xarray.assign(x=(x0_xarray.dims, xa.T))
