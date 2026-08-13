"""Class for 3D Var Data Assimilation Cycler object"""

import numpy as np
import jax.numpy as jnp
import jax
import jax.scipy as jscipy
import xarray as xr
from dabench import _xarray_jax as xj
from typing import Callable

from dabench import dacycler
import dabench.dacycler._utils as dac_utils
from dabench.model import Model

# For typing
ArrayLike = np.ndarray | jax.Array
XarrayDatasetLike = xr.Dataset | xj.XjDataset

class Var3D(dacycler.DACycler):
    """3D-Var DA Cycler

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
                 fgat: bool = False,
                 analysis_time_index: int | str = "mid",
                 fgat_representativeness: bool = True,
                 ):
        # 3D-Var-FGAT opt-in.  When True the analysis assimilates observations
        # distributed across the window: the innovation is formed against the
        # background trajectory AT each obs's true time, while the increment is
        # solved once at the single analysis time ``tau`` with the static B
        # (textbook 3D-Var-FGAT).  Default False keeps the legacy single-slice
        # 3D-Var path (byte-identical).
        if fgat:
            self._fgat = True
        self._analysis_time_spec = analysis_time_index
        self.fgat_representativeness = bool(fgat_representativeness)

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

        ana_ds = xb_ds.assign(x=(xb_ds.dims, xa.T))
        if not self._return_metrics:
            return ana_ds
        # Legacy single-slice 3D-Var: all in-window obs treated as concurrent
        # at the analysis time, so O-F/O-A are scored at that single slice and
        # end-of-window O-A is not separable (NaN, matching the 3D ensemble
        # path).  The spread metrics are B-derived (deterministic analogue of
        # the ensemble spread): ``obs_space_spread_background`` = obs-space std
        # of the static B; ``obs_space_spread_analysis_end`` = obs-space std of
        # the static posterior A = (B^-1 + H^T R^-1 H)^-1 (no propagation).
        dtype = xa.dtype
        y_flat = jnp.asarray(y, dtype)
        H = jnp.asarray(H, dtype)
        B = jnp.asarray(B, dtype)
        Hxb = H @ jnp.asarray(xb, dtype)
        Hxa = H @ jnp.asarray(xa, dtype)
        active = (jnp.asarray(obs_time_mask, bool).reshape(-1)
                  & jnp.asarray(obs_loc_mask, bool).reshape(-1))
        sigma2_diag = jnp.diag(jnp.asarray(R, dtype))
        Rinv = jnp.linalg.inv(jnp.asarray(R, dtype))
        A_post = jnp.linalg.inv(jnp.linalg.inv(B) + H.T @ Rinv @ H)
        spread_bg = dac_utils._b_derived_obs_spread(H, B, active, dtype=dtype)
        spread_ana = dac_utils._b_derived_obs_spread(
                H, A_post, active, dtype=dtype)
        metrics = dac_utils._obs_space_metrics(
                y_flat, Hxb, Hxa, active, sigma2_diag, ens_obs=None,
                return_per_obs=(self._metrics_mode == "debug"), dtype=dtype,
                spread_background_override=spread_bg,
                spread_analysis_end_override=spread_ana)
        return ana_ds, metrics

    def _fgat_solve(self,
                    xb_tau: ArrayLike,
                    d: ArrayLike,
                    H: ArrayLike,
                    R: ArrayLike,
                    B: ArrayLike) -> jax.Array:
        """3D-Var increment at ``tau`` from a precomputed FGAT innovation ``d``.

        Solves the B-preconditioned system
        ``(I + B H^T R^{-1} H) xa = xb_tau + B H^T R^{-1} (d + H xb_tau)``,
        whose solution is ``xa = xb_tau + B H^T (H B H^T + R)^{-1} d`` -- the
        3D-Var analysis with the innovation ``d`` (formed at the true obs
        times) and the increment built from the single-time background at
        ``tau`` (textbook 3D-Var-FGAT).
        """
        Rinv = jnp.linalg.inv(R)
        I = jnp.identity(xb_tau.size)
        BHt = jnp.dot(B, H.T)
        BHtRinv = jnp.dot(BHt, Rinv)
        A = I + jnp.dot(BHtRinv, H)
        y_eff = d + jnp.dot(H, xb_tau)
        b1 = xb_tau + jnp.dot(BHtRinv, y_eff)
        xa, _ = jscipy.sparse.linalg.cg(A, b1, x0=xb_tau, tol=1e-05,
                                        maxiter=1000)
        return xa

    def _cycle_and_forecast_fgat(self,
                                 cur_state: xj.XjDataset,
                                 filtered_idx: ArrayLike
                                 ) -> tuple[xj.XjDataset, XarrayDatasetLike]:
        """3D-Var-FGAT scan step: innovation at obs times, increment at tau."""
        cur_state = cur_state.to_xarray()
        cur_time = cur_state['_cur_time'].data
        cur_state = cur_state.drop_vars(['_cur_time'])
        obs_time_mask = filtered_idx > 0
        filtered_idx = filtered_idx - 1

        cur_obs_vals = jnp.array(
                self._obs_vector[self._observed_vars]
                .to_stacked_array('system', ['time']).data
                ).at[filtered_idx].get()
        cur_obs_times = jnp.array(
                self._obs_vector.time.data).at[filtered_idx].get()
        cur_obs_loc_indices = jnp.array(
                self._obs_vector.system_index.data
                ).at[:, filtered_idx].get().reshape(filtered_idx.shape[0], -1)
        cur_obs_loc_mask = jnp.array(self._obs_loc_masks).at[
                :, filtered_idx].get().astype(bool).reshape(
                filtered_idx.shape[0], -1)

        obs_window_indices = jnp.array([
                jnp.argmin(
                    jnp.abs(obs_time - (cur_time + self._model_timesteps))
                    ) for obs_time in cur_obs_times
            ])

        # Background trajectory across the window.
        _, forecast_states = self._step_forecast(
                cur_state, n_steps=self.steps_per_window)
        Xtraj = jnp.asarray(
                forecast_states.to_stacked_array('system', ['time']).data)

        tau = self._resolve_analysis_index()

        # Per-obs-time selection operator, then fold the obs-time / obs-location
        # masks into H (masked rows contribute nothing).
        n_times = cur_obs_loc_indices.shape[0]
        obs_dim = cur_obs_loc_indices.shape[1]
        Hs = jnp.zeros((n_times, obs_dim, self.system_dim))
        for i in range(n_times):
            Hs = Hs.at[i, jnp.arange(obs_dim), cur_obs_loc_indices[i]].set(1.0)
        time_mask = jnp.asarray(obs_time_mask, dtype=Hs.dtype)[:, None, None]
        loc_mask = jnp.asarray(cur_obs_loc_mask, dtype=Hs.dtype)[:, :, None]
        Hs = Hs * time_mask * loc_mask
        H = Hs.reshape(n_times * obs_dim, self.system_dim)

        # FGAT innovation: d = y - H x^f(t_obs), stacked over the window.
        def _hx_tobs(i):
            return Hs[i] @ Xtraj[obs_window_indices[i]]
        Hx_tobs = jax.vmap(_hx_tobs)(jnp.arange(n_times)).reshape(-1)
        y = jnp.asarray(cur_obs_vals).reshape(-1)
        d = y - Hx_tobs

        # Representativeness inflation: scale R by the number of distinct active
        # obs times folded onto the single-time increment.
        n_obs_times = jnp.sum(jnp.asarray(obs_time_mask, dtype=Hs.dtype))
        repr_factor = (jnp.maximum(n_obs_times, 1.0)
                       if self.fgat_representativeness else 1.0)
        if self.R is None:
            R = self._calc_default_R(y, self.obs_error_sd) * repr_factor
        else:
            R = self.R * repr_factor
        B = self._calc_default_B() if self.B is None else self.B

        xb_tau = Xtraj[tau]
        xa = self._fgat_solve(xb_tau, d, H, R, B)

        xdims = cur_state['x'].dims
        ana_tau = cur_state.assign(x=(xdims, xa.T))
        if tau == self.steps_per_window - 1:
            next_state = ana_tau
            xa_end = xa
        else:
            next_state, end_states = self._step_forecast(
                    ana_tau, n_steps=self.steps_per_window - tau)
            xa_end = jnp.asarray(
                    end_states.to_stacked_array('system', ['time']).data)[-1]
        next_state = next_state.assign(
            _cur_time=cur_time + self.analysis_window
            ).assign_coords(
                cur_state.coords).assign_attrs(cur_state.attrs)

        if self._return_metrics:
            metrics = self._obs_metrics_fgat(
                    Xtraj, xa, xa_end, tau, Hs, y, R,
                    obs_window_indices, obs_time_mask, cur_obs_loc_mask,
                    H, B)
            return xj.from_xarray(next_state), (forecast_states, metrics)
        return xj.from_xarray(next_state), forecast_states

    def _obs_metrics_fgat(self, Xtraj, xa, xa_end, tau, Hs, y, R,
                          obs_window_indices, obs_time_mask, cur_obs_loc_mask,
                          H, B):
        """Obs-space metrics for the deterministic 3D-Var-FGAT path.

        O-F is the FGAT innovation ``y_i - H_i x^f(t_obs_i)`` scored at each
        obs's APPROPRIATE time; O-A is ``y_i - H_i x^a(tau)`` (the analysis is
        strictly valid only at ``tau``).  A tau-restricted O-F/O-A pair (obs at
        ``tau`` only) gives the like-for-like fit where O-A <= O-F holds, and
        the end-of-window O-A scores the next-cycle IC (analysis propagated to
        the window end vs obs valid at the end).  The spread metrics are
        B-derived (static-B analogue of the ensemble spread; NO propagation --
        3D-Var-FGAT carries no covariance dynamics): background is the obs-space
        std of the static B over the assimilated window obs, and
        ``obs_space_spread_analysis_end`` is the obs-space std of the static
        posterior A = (B^-1 + H^T R^-1 H)^-1 at ``tau`` (same stacked H / window
        R used by the FGAT increment).
        """
        dtype = xa.dtype
        n_times = Hs.shape[0]
        obs_dim = Hs.shape[1]
        y = jnp.asarray(y, dtype).reshape(-1)
        idx = jnp.arange(n_times)

        yf_bar = jax.vmap(
                lambda i: Hs[i] @ Xtraj[obs_window_indices[i]])(
                idx).reshape(-1).astype(dtype)
        ya_bar = jax.vmap(lambda i: Hs[i] @ xa)(idx).reshape(-1).astype(dtype)
        ya_end = jax.vmap(lambda i: Hs[i] @ xa_end)(
                idx).reshape(-1).astype(dtype)

        active = (jnp.repeat(jnp.asarray(obs_time_mask, bool), obs_dim)
                  & jnp.asarray(cur_obs_loc_mask, bool).reshape(-1))
        # R is the full (n_times*obs_dim) window covariance, so its diagonal is
        # already the per-flat-obs variance (repr-inflated).
        sigma2_diag = jnp.diag(jnp.asarray(R, dtype)).reshape(-1)

        end_idx = self.steps_per_window - 1
        owi = jnp.asarray(obs_window_indices)
        end_active = active & jnp.repeat(owi == end_idx, obs_dim)
        tau_active = active & jnp.repeat(owi == tau, obs_dim)

        # B-derived spreads (static-B analogue; no propagation).  Uses the same
        # stacked H / window R the FGAT increment was built from.
        H = jnp.asarray(H, dtype)
        B = jnp.asarray(B, dtype)
        Rinv = jnp.linalg.inv(jnp.asarray(R, dtype))
        A_post = jnp.linalg.inv(jnp.linalg.inv(B) + H.T @ Rinv @ H)
        spread_bg = dac_utils._b_derived_obs_spread(H, B, active, dtype=dtype)
        spread_ana = dac_utils._b_derived_obs_spread(
                H, A_post, active, dtype=dtype)

        cmp = dac_utils._obs_space_metrics(
                y, yf_bar, ya_bar, tau_active, sigma2_diag, dtype=dtype)
        out = dac_utils._obs_space_metrics(
                y, yf_bar, ya_bar, active, sigma2_diag, ens_obs=None,
                return_per_obs=(self._metrics_mode == "debug"), dtype=dtype,
                Hxa_end_mean=ya_end, end_active_mask=end_active,
                spread_background_override=spread_bg,
                spread_analysis_end_override=spread_ana)
        out["o_minus_f_rms_at_tau"] = cmp["o_minus_f_rms"]
        out["o_minus_a_rms_at_tau"] = cmp["o_minus_a_rms"]
        return out
