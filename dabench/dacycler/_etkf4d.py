"""Class for 4D Ensemble Transform Kalman Filter (4D-ETKF) DA Cycler"""

import numpy as np
import jax
import jax.numpy as jnp
from jax.scipy import linalg
import xarray as xr
from dabench import _xarray_jax as xj
from typing import Callable

from dabench.dacycler import ETKF


# For typing
ArrayLike = np.ndarray | jax.Array
XarrayDatasetLike = xr.Dataset | xj.XjDataset


class ETKF4D(ETKF):
    """4D Ensemble Transform Kalman Filter DA Cycler.

    The 4D extension of :class:`ETKF`.  Where :class:`ETKF` assimilates a
    single observation slice per analysis window, :class:`ETKF4D` rides the
    base cycler's ``_in_4d`` plumbing (:meth:`DACycler._cycle_and_forecast_4d`)
    and assimilates observations distributed *across* the window: each
    ensemble member is rolled the full window, mapped to observation space at
    the model timestep nearest each observation time, and the ensemble-space
    transform is built from the window-stacked observation-minus-forecast
    innovations.

    Like :class:`ETKF`, the analysis runs entirely in model space and uses the
    linear observation operator ``H`` (or the default selector built from the
    observation location indices) to map into observation space for the
    innovation; no tangent-linear model is required.

    The transform weights ``(W_a, w_a)`` are time-invariant ensemble-space
    objects.  This cycler applies them to the *window-start* ensemble and
    returns that as the analysis; the base cycler then re-forecasts it across
    the window (the "initial-time" / 4D-Var-like correction the user described
    -- the corrected initial ensemble is propagated nonlinearly).  Applying the
    same weights to the window-*end* perturbations instead (the "final-time" /
    filter correction that skips the re-forecast) would require overriding
    :meth:`DACycler._cycle_and_forecast_4d`; it is intentionally left out here.

    Args:
        system_dim: System dimension.
        delta_t: The timestep of the model (assumed uniform).
        model_obj: Forecast model object.
        B: Unused (kept for the :class:`ETKF` signature); the background error
            covariance is the flow-dependent ensemble covariance.
        R: Observation error covariance matrix. If None, built as a diagonal
            from ``obs_error_sd``.
        H: Observation operator with shape (obs_dim, system_dim). If not
            provided, a default selector is built per obs time from the
            observation location indices.
        h: Optional callable observation operator (unsupported: only linear
            ``H`` paths are implemented, matching :class:`ETKF`).
        ensemble_dim: Number of ensemble members. Default 4.
        multiplicative_inflation: Ensemble-deviation inflation. Default 1.0.
    """
    _in_4d: bool = True
    _uses_ensemble: bool = True

    def _calc_default_H(self,
                        obs_loc_indices: ArrayLike
                        ) -> jax.Array:
        """Per-obs-time selection operator, shape (n_times, obs_dim, sys)."""
        Hs = jnp.zeros((obs_loc_indices.shape[0], obs_loc_indices.shape[1],
                        self.system_dim))
        for i in range(Hs.shape[0]):
            Hs = Hs.at[i, jnp.arange(Hs.shape[1]), obs_loc_indices[i]].set(1.0)
        return Hs

    def _compute_analysis_4d(self,
                             Xb: ArrayLike,
                             Yb: ArrayLike,
                             Y: ArrayLike,
                             rinv_diag: ArrayLike,
                             rho: float = 1.0
                             ) -> ArrayLike:
        """ETKF transform from window-stacked obs, applied to ``Xb``.

        Mirrors :meth:`ETKF._compute_analysis` but takes the
        observation-space ensemble ``Yb`` (stacked over all obs times in the
        window) and a diagonal ``R^{-1}`` carrying the obs-time / obs-location
        masks, rather than recomputing ``Yb`` from a single-time ``Xb``.

        Args:
            Xb: Window-start background ensemble, shape (system_dim, ens_dim).
            Yb: Obs-space ensemble across the window, shape (n_obs, ens_dim).
            Y: Flattened observation vector, shape (n_obs,).
            rinv_diag: Diagonal of masked ``R^{-1}``, shape (n_obs,).
            rho: Multiplicative inflation factor (1.0 = none).

        Returns:
            Xa: Analysis ensemble, shape (system_dim, ens_dim).
        """
        system_dim, ensemble_dim = Xb.shape
        U = jnp.ones((ensemble_dim, ensemble_dim)) / ensemble_dim
        I = jnp.identity(ensemble_dim)

        Xb_pert = Xb @ (I - U)
        Xb = Xb_pert + Xb @ U
        Xb_bar = jnp.mean(Xb, axis=1)
        Xb_pert = Xb @ (I - U)

        yb_bar = jnp.mean(Yb, axis=1)
        Yb_pert = Yb @ (I - U)

        # Diagonal R^{-1} (masks fold in as zeroed entries), so the obs term
        # is (Yb_pert^T R^{-1}) acting on Yb_pert / the innovation.
        YtRinv = Yb_pert.T * rinv_diag[None, :]
        Pa_ens = jnp.linalg.pinv((ensemble_dim - 1) / rho * I
                                 + YtRinv @ Yb_pert,
                                 rtol=1e-15)
        Wa = linalg.sqrtm((ensemble_dim - 1) * Pa_ens).real
        wa = Pa_ens @ (YtRinv @ (Y - yb_bar))

        Xa_pert = Xb_pert @ Wa
        Xa_bar = Xb_bar + jnp.ravel(Xb_pert @ wa)
        v = jnp.ones((1, ensemble_dim))
        Xa = Xa_pert + Xa_bar[:, None] @ v
        return Xa

    def _cycle_obsop(self,
                     xb0_ds: XarrayDatasetLike,
                     obs_values: ArrayLike,
                     obs_loc_indices: ArrayLike,
                     obs_time_mask: ArrayLike,
                     obs_loc_mask: ArrayLike,
                     H: ArrayLike | None = None,
                     h: Callable | None = None,
                     R: ArrayLike | None = None,
                     B: ArrayLike | None = None,
                     obs_window_indices: ArrayLike | None = None,
                     ) -> XarrayDatasetLike:
        # 1. Per-obs-time observation operator Hs: (n_times, obs_dim, sys)
        if self.H is None:
            Hs = self._calc_default_H(obs_loc_indices)
        else:
            Hs = jnp.repeat(jnp.asarray(self.H)[jnp.newaxis],
                            obs_values.shape[0], axis=0)
        n_times, obs_dim = Hs.shape[0], Hs.shape[1]

        # 2. Masked diagonal R^{-1} (obs-time mask x obs-location mask).
        sigma2 = jnp.asarray(self.obs_error_sd, dtype=Hs.dtype) ** 2
        time_mask = jnp.repeat(jnp.asarray(obs_time_mask, dtype=Hs.dtype),
                               obs_dim)
        loc_mask = jnp.asarray(obs_loc_mask, dtype=Hs.dtype).reshape(-1)
        rinv_diag = (time_mask * loc_mask) / sigma2

        # 3. Roll the ensemble across the window, gather obs-space predictions
        #    at the model timestep nearest each observation time.
        _, fc = self._step_forecast(xb0_ds, n_steps=self.steps_per_window)
        Xtraj = jnp.asarray(
                fc.to_stacked_array('system', ['ensemble', 'time']).data)
        owi = jnp.asarray(obs_window_indices)

        def _obs_pred(i):
            return Hs[i] @ Xtraj[:, owi[i], :].T          # (obs_dim, ens)

        Yb = jax.vmap(_obs_pred)(jnp.arange(n_times)).reshape(
                -1, self.ensemble_dim)

        Xb0 = jnp.asarray(
                xb0_ds.to_stacked_array('system', ['ensemble']).data).T

        Xa = self._compute_analysis_4d(
                Xb=Xb0, Yb=Yb, Y=obs_values.reshape(-1), rinv_diag=rinv_diag,
                rho=self.multiplicative_inflation)

        return xb0_ds.assign(x=(['ensemble', 'i'], Xa.T))
