"""Class for 4D Local Ensemble Transform Kalman Filter (4D-LETKF) DA Cycler."""

import numpy as np
import jax
import jax.numpy as jnp
import xarray as xr
from dabench import _xarray_jax as xj
from typing import Callable

from dabench.dacycler import ETKF4D
from dabench.dacycler._letkf import LETKF


# For typing
ArrayLike = np.ndarray | jax.Array
XarrayDatasetLike = xr.Dataset | xj.XjDataset


class LETKF4D(LETKF, ETKF4D):
    """4D Local Ensemble Transform Kalman Filter DA Cycler.

    The 4D extension of :class:`LETKF`, standing to :class:`LETKF` exactly as
    :class:`~dabench.dacycler.ETKF4D` stands to :class:`~dabench.dacycler.ETKF`.
    It rides :class:`ETKF4D`'s ``_in_4d`` window plumbing (each member rolled
    the full window, mapped to obs space at the model step nearest each obs
    time, window-stacked innovations) but replaces the *global* transform with
    :class:`LETKF`'s domain-localized per-gridpoint solve.

    Flow (per window):

    1. Roll the incoming ensemble the full window (the scored prior trajectory).
    2. Window-stack ``(Yb, Y, rinv_diag)`` from the innovations (obs-space is
       already grid-space; ``H`` unchanged) via :meth:`ETKF4D._build_yb`.
    3. Lift the background at the analysis time ``tau`` to grid, run the fused
       per-gridpoint local ETKF solve on the window-stacked innovations, SHT
       back to spectral, and apply relaxation/additive on the spectral
       perturbations (:meth:`LETKF._localized_analysis`).
    4. Propagate the analysis ensemble to the window end for the next IC.

    Args mirror :class:`LETKF` (localization config) and :class:`ETKF4D`
    (``analysis_time_index``).
    """
    _in_4d: bool = True
    _uses_ensemble: bool = True

    def _cycle_and_forecast_4d(self,
                               cur_state: xj.XjDataset,
                               filtered_idx: ArrayLike
                               ) -> tuple[xj.XjDataset, XarrayDatasetLike]:
        # 1. Get data; restore filtered_idx and obs-time mask (as in ETKF4D).
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

        # Closest model timesteps that match each obs time.
        obs_window_indices = jnp.array([
                jnp.argmin(
                    jnp.abs(obs_time - (cur_time + self._model_timesteps))
                    ) for obs_time in cur_obs_times
            ])

        # 2. Prior forecast of the incoming ensemble across the window; this is
        #    the trajectory scored for the cycle (filter, not smoother).
        _, forecast_states = self._step_forecast(
                cur_state, n_steps=self.steps_per_window)

        # 3. Window-stacked innovations (obs-space already grid-space).
        Xtraj, Yb, Y, rinv_diag = self._build_yb(
                forecast_states, cur_obs_vals, cur_obs_loc_indices,
                obs_time_mask, cur_obs_loc_mask, obs_window_indices)

        # Per-obs grid indices for the taper: flatten the window-stacked
        # (n_times, obs_dim) location indices to match the (n_obs,) obs axis.
        obs_loc_flat = jnp.asarray(cur_obs_loc_indices).reshape(-1)

        # 4. Localized analysis at the chosen in-window time tau.
        tau = self._resolve_analysis_index()
        add_key = (jax.random.fold_in(
                    self._additive_key,
                    jnp.round(cur_time / self.analysis_window).astype(jnp.int32))
                   if self.additive_inflation > 0.0 else None)
        Xb_tau = Xtraj[:, tau, :].T                            # (system, ens)
        Xa = self._localized_analysis(
                Xb_tau, Yb, Y, rinv_diag, obs_loc_flat,
                rho=self.multiplicative_inflation, key=add_key)

        # 5. Reuse the incoming state's dim names (ensemble-first carry).
        xdims = cur_state['x'].dims
        Xa_oriented = Xa.T if xdims[0] == 'ensemble' else Xa
        ana_tau = cur_state.assign(x=(xdims, Xa_oriented))
        if tau == self.steps_per_window - 1:
            next_state = ana_tau
        else:
            next_state, _ = self._step_forecast(
                    ana_tau, n_steps=self.steps_per_window - tau)

        next_state = next_state.assign(
            _cur_time=cur_time + self.analysis_window
            ).assign_coords(
                cur_state.coords).assign_attrs(cur_state.attrs)

        return xj.from_xarray(next_state), forecast_states
