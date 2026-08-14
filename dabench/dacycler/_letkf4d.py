"""Class for 4D Local Ensemble Transform Kalman Filter (4D-LETKF) DA Cycler."""

import numpy as np
import jax
import jax.numpy as jnp
import xarray as xr
from dabench import _xarray_jax as xj
from typing import Callable

from dabench.dacycler import ETKF4D
from dabench.dacycler._letkf import LETKF
import dabench.dacycler._utils as dac_utils


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

    def __init__(self, *args, analysis_time_index: int | str = "end",
                 **kwargs):
        # 4D default placement is the window END (filter placement), matching
        # :class:`ETKF4D`.  Pinned explicitly here rather than relying on the
        # MRO reaching ``ETKF4D.__init__`` before ``ETKF.__init__`` (whose 3D
        # default is "mid"), so the 4D default cannot silently regress.
        super().__init__(*args, analysis_time_index=analysis_time_index,
                         **kwargs)

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

        metrics = None
        if self._return_metrics:
            dtype = Xtraj.dtype
            yb_bar = jnp.mean(Yb, axis=1)
            if self.H is None:
                Hs = self._calc_default_H(cur_obs_loc_indices)
            else:
                Hs = jnp.repeat(jnp.asarray(self.H)[jnp.newaxis],
                                cur_obs_vals.shape[0], axis=0)

            # Localized analysis-at-time closure; the shared scorer selects the
            # causal (tau) vs time-matched (window-start) placement per
            # oa_score_mode.
            def _analysis_at(t):
                return self._localized_analysis(
                        Xtraj[:, t, :].T, Yb, Y, rinv_diag, obs_loc_flat,
                        rho=self.multiplicative_inflation, key=None)

            ya_bar = self._score_oa_ya(
                    cur_state, _analysis_at, tau, Hs, obs_window_indices, dtype)
            active = rinv_diag > 0
            sigma2_diag = jnp.where(
                    active,
                    1.0 / jnp.where(active, rinv_diag,
                                    jnp.ones_like(rinv_diag)),
                    jnp.zeros_like(rinv_diag))
            ens_obs = Yb - yb_bar[:, None]

            # End-of-window O-A (next-cycle IC quality): localized analysis at
            # tau, propagated to the window END, scored against end-valid obs.
            # The full end ENSEMBLE also gives the next-cycle IC spread.
            end_idx = self.steps_per_window - 1
            Xa_end_ens = self._xa_end_ensemble(
                    cur_state, _analysis_at(tau), tau)      # (system, ens)
            Xa_end = jnp.mean(Xa_end_ens, axis=1)
            ya_end = jax.vmap(lambda i: Hs[i] @ Xa_end)(
                    jnp.arange(Hs.shape[0])).reshape(-1).astype(dtype)
            # End-of-window analysis obs-space perturbations (obs x ens): drives
            # ``obs_space_spread_analysis_end`` (spread of the ensemble handed
            # to the next cycle -- NOT ``obs_space_spread_background`` in-window).
            Ya_end = jax.vmap(lambda i: Hs[i] @ Xa_end_ens)(
                    jnp.arange(Hs.shape[0])).reshape(-1, self.ensemble_dim)
            ens_obs_end = (Ya_end
                           - jnp.mean(Ya_end, axis=1)[:, None]).astype(dtype)
            owi = jnp.asarray(obs_window_indices)
            end_active = active & (jnp.repeat(owi, Hs.shape[1]) == end_idx)
            metrics = dac_utils._obs_space_metrics(
                    jnp.asarray(Y, dtype), yb_bar.astype(dtype),
                    ya_bar, active, sigma2_diag, ens_obs=ens_obs,
                    return_per_obs=(self._metrics_mode == "debug"),
                    dtype=dtype,
                    Hxa_end_mean=ya_end, end_active_mask=end_active,
                    ens_obs_end=ens_obs_end)
            return xj.from_xarray(next_state), (forecast_states, metrics)
        return xj.from_xarray(next_state), forecast_states

    def capture_first_transforms(self,
                                 input_state: XarrayDatasetLike,
                                 start_time: float,
                                 obs_vector: XarrayDatasetLike,
                                 n_cycles: int,
                                 obs_error_sd=None,
                                 analysis_window: float = 0.2,
                                 analysis_time_in_window: float | None = None
                                 ) -> np.ndarray:
        """Materialize the real per-gridpoint transforms ``A`` from cycle 0.

        Runs the SAME pre-scan setup (:meth:`_prepare_cycle`) and the SAME
        cycle-0 obs gather / window forecast / window-stacking as
        :meth:`_cycle_and_forecast_4d`, but EAGERLY (outside ``lax.scan``) and
        stops at :meth:`LETKF.capture_A_matrices` -- so the returned
        ``(grid_dim, K, K)`` stack is EXACTLY the SPD transforms the first
        analysis would solve, as concrete host arrays.  For offline solver /
        precision / ridge diagnostics; does not run the analysis or advance the
        filter.

        Args mirror :meth:`~dabench.dacycler.DACycler.cycle`.

        Returns:
            The cycle-0 per-gridpoint transform stack ``(grid_dim, K, K)`` as a
            NumPy array (dtype follows the ensemble precision).
        """
        input_state, all_filtered_padded = self._prepare_cycle(
            input_state, start_time, obs_vector, obs_error_sd, n_cycles,
            analysis_window, analysis_time_in_window)

        # Cycle-0 obs indices (first padded row); reproduce the body's gather.
        cur_state = input_state
        cur_time = jnp.asarray(cur_state['_cur_time'].data)
        cur_state = cur_state.drop_vars(['_cur_time'])
        filtered_idx = jnp.asarray(all_filtered_padded[0])
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

        _, forecast_states = self._step_forecast(
                cur_state, n_steps=self.steps_per_window)
        Xtraj, Yb, Y, rinv_diag = self._build_yb(
                forecast_states, cur_obs_vals, cur_obs_loc_indices,
                obs_time_mask, cur_obs_loc_mask, obs_window_indices)
        obs_loc_flat = jnp.asarray(cur_obs_loc_indices).reshape(-1)

        tau = self._resolve_analysis_index()
        Xb_tau = Xtraj[:, tau, :].T                            # (system, ens)
        A_stack = self.capture_A_matrices(
                Xb_tau, Yb, Y, rinv_diag, obs_loc_flat,
                rho=self.multiplicative_inflation)
        return np.asarray(A_stack)
