"""Class for 3D Var Data Assimilation Cycler object"""

import numpy as np
import jax.numpy as jnp
import jax
import jax.scipy as jscipy
import xarray as xr
import xarray_jax as xj
from typing import Callable

from dabench import dacycler
from dabench.model import Model

# For typing
ArrayLike = np.ndarray | jax.Array
XarrayDatasetLike = xr.Dataset | xj.XjDataset

class Var3D(dacycler.DACycler):
    """Class for building 3DVar DA Cycler

    Args:
        system_dim: System dimension.
        delta_t: The timestep of the model (assumed uniform)
        model_obj: Forecast model object.
        B: Initial / static background error covariance. Shape:
            (system_dim, system_dim). If not provided, will be calculated
            automatically.
        R: Observation error covariance matrix. Shape
            (obs_dim, obs_dim). If not provided, will be calculated
            automatically.
        H: Observation operator with shape: (obs_dim, system_dim).
            If not provided will be calculated automatically.
        h: Optional observation operator as function. More flexible
            (allows for more complex observation operator). Default is None.
        """
    _in_4d: bool = False
    _uses_ensemble: bool = False

    def __init__(self,
                 system_dim: int,
                 delta_t: float,
                 model_obj: Model,
                 B: ArrayLike | None = None,
                 R: ArrayLike | None = None,
                 H: ArrayLike | None = None,
                 h: Callable | None = None,
                 ):

        super().__init__(system_dim=system_dim,
                         delta_t=delta_t,
                         model_obj=model_obj,
                         B=B, R=R, H=H, h=h)

    def _cycle_obsop(self,
                     xb_ds: XarrayDatasetLike,
                     obs_values: ArrayLike,
                     obs_loc_indices: ArrayLike,
                     obs_time_mask: ArrayLike,
                     obs_loc_mask: ArrayLike,
                     H: ArrayLike,
                     h: Callable | None = None,
                     R: ArrayLike | None = None,
                     B: ArrayLike | None = None) -> XarrayDatasetLike:
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

        xb = xb_ds.to_stacked_array('system',[]).data.flatten()
        y = obs_values.flatten()

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
        b1 = xb + jnp.dot(BHtRinv, y)

        # Use minimization algorithm to minimize cost function:
        xa, ierr = jscipy.sparse.linalg.cg(A, b1, x0=xb, tol=1e-05,
                                           maxiter=1000)

        return xb_ds.assign(x=(xb_ds.dims, xa.T))
