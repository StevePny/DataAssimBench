"""Class for Ensemble Transform Kalman Filter (ETKF) DA Class"""

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

class ETKF(dacycler.DACycler):
    """Class for building ETKF DA Cycler

    Attributes:
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
        ensemble_dim: Number of ensemble instances for ETKF. Default is
            4. Higher ensemble_dim increases accuracy but has performance cost.
        multiplicative_inflation: Scaling factor by which to multiply ensemble
            deviation. Default is 1.0 (no inflation).
    """

    def __init__(self,
                 system_dim: int,
                 delta_t: float,
                 model_obj: Model,
                 B: ArrayLike | None = None,
                 R: ArrayLike | None = None,
                 H: ArrayLike | None = None,
                 h: Callable | None = None,
                 ensemble_dim: int = 4,
                 multiplicative_inflation: float = 1.0
                 ):

        self.ensemble_dim = ensemble_dim
        self.multiplicative_inflation = multiplicative_inflation

        super().__init__(system_dim=system_dim,
                         delta_t=delta_t,
                         model_obj=model_obj,
                         in_4d=False,
                         ensemble=True,
                         B=B, R=R, H=H, h=h)

    def _step_forecast(self,
                       Xa: XarrayDatasetLike,
                       n_steps: int = 1
                       ) -> XarrayDatasetLike:
        """Ensemble method needs a slightly different _step_forecast method"""
        ensemble_forecasts = []
        ensemble_inputs = []
        for i in range(self.ensemble_dim):
            cur_inputs, cur_forecast = self.model_obj.forecast(
                    Xa.isel(ensemble=i),
                    n_steps=n_steps
                    )
            ensemble_inputs.append(cur_inputs)
            ensemble_forecasts.append(cur_forecast)

        return (xr.concat(ensemble_inputs, dim='ensemble'),
                xr.concat(ensemble_forecasts, dim='ensemble'))

    def _apply_obsop(self,
                     Xb: ArrayLike,
                     H: ArrayLike | None,
                     h: Callable | None
                     ) -> ArrayLike:
        if H is not None:
            Yb = H @ Xb
        else:
            Yb = h(Xb)

        return Yb

    def _compute_analysis(self,
                          Xb: ArrayLike,
                          Y: ArrayLike,
                          H: ArrayLike | None,
                          h: Callable | None,
                          R: ArrayLike,
                          rho: float = 1.0
                          ) ->  ArrayLike:
        """ETKF analysis algorithm

        Args:
          Xb: Forecast/background ensemble with shape
            (system_dim, ensemble_dim).
          Y: Observation array with shape (obs_time_time, observation_dim)
          H: Linear observation operator with shape (observation_dim,
            system_dim).
          h: Callable observation operator (optional).
          R: Observation error covariance matrix with shape
            (observation_dim, observation_dim)
          rho: Multiplicative inflation factor. Default=1.0,
            (i.e. no inflation)

        Returns:
          Xa: Analysis ensemble [size: (system_dim, ensemble_dim)]
        """
        # Number of state variables, ensemble members and observations
        system_dim, ensemble_dim = Xb.shape

        # Auxiliary matrices that will ease the computations
        U = jnp.ones((ensemble_dim, ensemble_dim))/ensemble_dim
        I = jnp.identity(ensemble_dim)

        # The ensemble is inflated (rho=1.0 is no inflation)
        Xb_pert = Xb @ (I-U)
        Xb = Xb_pert + Xb @ U

        # Map every ensemble member into observation space
        Yb = self._apply_obsop(Xb, H, h)

        # Get ensemble means and perturbations
        Xb_bar = jnp.mean(Xb,  axis=1)
        Xb_pert = Xb @ (I-U)

        yb_bar = jnp.mean(Yb, axis=1)
        Yb_pert = Yb @ (I-U)

        # Compute the analysis
        if len(R) > 0:
            Rinv = jnp.linalg.pinv(R, rtol=1e-15)

            Pa_ens = jnp.linalg.pinv((ensemble_dim-1)/rho*I
                                     + Yb_pert.T @ Rinv @ Yb_pert,
                                     rtol=1e-15)
            Wa = linalg.sqrtm((ensemble_dim-1) * Pa_ens)
            Wa = Wa.real
        else:
            Rinv = jnp.zeros_like(R, dtype=R.dtype)
            Pa_ens = jnp.zeros((ensemble_dim, ensemble_dim), dtype=R.dtype)
            Wa = jnp.zeros((ensemble_dim, ensemble_dim), dtype=R.dtype)

        wa = Pa_ens @ Yb_pert.T @ Rinv @ (Y.flatten()-yb_bar)

        Xa_pert = Xb_pert @ Wa

        Xa_bar = Xb_bar + jnp.ravel(Xb_pert @ wa)

        v = jnp.ones((1, ensemble_dim))
        Xa = Xa_pert + Xa_bar[:, None] @ v

        return Xa

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

        # Analysis cycles over all obs in data_obs
        Xa = self._compute_analysis(Xb=Xb,
                                    Y=obs_values,
                                    H=H,
                                    h=h,
                                    R=R,
                                    rho=self.multiplicative_inflation)

        return Xb_ds.assign(x=(['ensemble','i'], Xa.T))
