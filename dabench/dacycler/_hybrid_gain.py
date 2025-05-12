"""Class for Hybrid Gain (ETKF + 3DVar) Data Assimilation"""

import numpy as np
import jax
import jax.numpy as jnp
from jax.scipy import linalg
import xarray as xr
import xarray_jax as xj
from typing import Callable

from dabench import dacycler
from dabench.model import Model


# For typing
ArrayLike = np.ndarray | jax.Array
XarrayDatasetLike = xr.Dataset | xj.XjDataset

class HybridGain(dacycler.DACycler):
    """HybridGain DA, combining ETKF with 3DVar

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
        alpha: Weight for 3DVar DA analysis. If 0.0, runs pure ETKF. If 1.0,
            runs pure 3DVar. Default is 0.2.
        ensemble_dim: Number of ensemble instances for ETKF. Default is
            4. Higher ensemble_dim increases accuracy but has performance cost.
        multiplicative_inflation: Scaling factor by which to multiply ensemble
            deviation. Default is 1.0 (no inflation).
    """
    _in_4d: bool = False
    _uses_ensemble: bool = True

    def __init__(self,
                 system_dim: int,
                 delta_t: float,
                 model_obj: Model,
                 B: ArrayLike | None = None,
                 R: ArrayLike | None = None,
                 H: ArrayLike | None = None,
                 h: Callable | None = None,
                 alpha: float = 0.2,
                 ensemble_dim: int = 4,
                 multiplicative_inflation: float = 1.0
                 ):

        self.ensemble_dim = ensemble_dim
        self.multiplicative_inflation = multiplicative_inflation
        self.alpha = alpha

        # Create ETKF DA Cycler
        self._etkf_da = dacycler.ETKF(
            system_dim=system_dim,
            delta_t=delta_t,
            model_obj=model_obj,
            R=R,
            H=H,
            h=h,
            ensemble_dim=ensemble_dim,
            multiplicative_inflation=multiplicative_inflation
        )
        # Create 3D-Var DA Cycler
        self._var3d_da = dacycler.Var3D(
            system_dim=system_dim,
            delta_t=delta_t,
            model_obj=model_obj,
            R=R,
            H=H,
            h=h,
            B=B
        )

        super().__init__(system_dim=system_dim,
                         delta_t=delta_t,
                         model_obj=model_obj,
                         B=B, R=R, H=H, h=h)

    def _step_forecast(self,
                       Xa: XarrayDatasetLike,
                       n_steps: int = 1
                       ) -> XarrayDatasetLike:
        """Ensemble method needs a slightly different _step_forecast method"""
        return self._etkf_da._step_forecast(Xa, n_steps)

    def _cycle_obsop(self,
                     Xb_ds: XarrayDatasetLike,
                     obs_values: ArrayLike,
                     obs_loc_indices: ArrayLike,
                     obs_time_mask: ArrayLike,
                     obs_loc_mask: ArrayLike,
                     H: ArrayLike | None = None,
                     h: Callable | None = None,
                     R: ArrayLike | None = None,
                     B: ArrayLike | None = None
                     ) -> XarrayDatasetLike:
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

        Xb = Xb_ds.to_stacked_array('system',['ensemble']).data.T
        n_sys, n_ens = Xb.shape
        assert n_ens == self.ensemble_dim, (
                'cycle:: model_forecast must have dimension {}x{}').format(
                    self.ensemble_dim, self.system_dim)

        # Apply obs masks to H
        H = jnp.where(obs_time_mask.flatten(), H.T, 0).T
        H = jnp.where(obs_loc_mask.flatten(), H.T, 0).T

        # Compute ETKF analysis
        Xa_etkf = self._etkf_da._compute_analysis(Xb=Xb,
                                    Y=obs_values,
                                    H=H,
                                    h=h,
                                    R=R,
                                    rho=self.multiplicative_inflation)

        # Compute Var3D Analysis
        xa_var3d = self._var3d_da._compute_analysis(xb=jnp.mean(Xb, axis=1).flatten(),
                                    y=obs_values.flatten(),
                                    H=H,
                                    B=B,
                                    Rinv=jnp.linalg.inv(R))

        xa_etkf_mean = jnp.mean(Xa_etkf, axis=1)
        xa_final = self.alpha*xa_var3d + (1-self.alpha)*xa_etkf_mean
        Xa_final = Xa_etkf.T - (xa_etkf_mean - xa_final)

        return Xb_ds.assign(x=(['ensemble','i'], Xa_final))
