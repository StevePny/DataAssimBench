"""Class for 4D Ensemble Transform Kalman Filter (4D-ETKF) DA Cycler"""

import numpy as np
import jax
import jax.numpy as jnp
import xarray as xr
from dabench import _xarray_jax as xj
from typing import Callable

from dabench.dacycler import ETKF
import dabench.dacycler._utils as dac_utils


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
    objects built from the window-stacked innovations.  They are applied to the
    background ensemble at a single in-window analysis time selected by
    ``analysis_time_index`` (default the window *end*, the filter placement);
    the resulting analysis ensemble is then propagated forward to the window
    end for every member, and that propagated ensemble is the initial condition
    for the next cycle.  The trajectory scored for each cycle is the *prior*
    forecast of the incoming ensemble across the window, so the analysis
    correction enters the score honestly at the next cycle (no future-obs
    smoothing of the current window).  Selecting ``analysis_time_index=0``
    ("start") recovers the 4D-Var-like correction where the weights act on the
    window-start ensemble which is then re-forecast across the window.

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
        analysis_time_index: In-window timestep at which the transform weights
            are applied. Accepts ``"start"`` (index 0), ``"mid"``, ``"end"``
            (default; index ``steps_per_window - 1``, the next-cycle IC time),
            or an integer index into ``[0, steps_per_window - 1]`` (negative
            indices count from the window end).
    """
    _in_4d: bool = True
    _uses_ensemble: bool = True

    def __init__(self, *args, analysis_time_index: int | str = "end",
                 **kwargs):
        # 4D default placement is the window END (filter placement); the shared
        # resolver lives on the base :class:`ETKF`.  FGAT is meaningless for the
        # 4D path (it already assimilates across the window), so it is not
        # forwarded here.
        super().__init__(*args, analysis_time_index=analysis_time_index,
                         **kwargs)

    def _calc_default_H(self,
                        obs_loc_indices: ArrayLike
                        ) -> jax.Array:
        """Per-obs-time selection operator, shape (n_times, obs_dim, sys)."""
        Hs = jnp.zeros((obs_loc_indices.shape[0], obs_loc_indices.shape[1],
                        self.system_dim))
        for i in range(Hs.shape[0]):
            Hs = Hs.at[i, jnp.arange(Hs.shape[1]), obs_loc_indices[i]].set(1.0)
        return Hs

    def _build_yb(self,
                  fc: XarrayDatasetLike,
                  obs_values: ArrayLike,
                  obs_loc_indices: ArrayLike,
                  obs_time_mask: ArrayLike,
                  obs_loc_mask: ArrayLike,
                  obs_window_indices: ArrayLike,
                  ) -> tuple[ArrayLike, ArrayLike, ArrayLike, ArrayLike]:
        """Map a window forecast into stacked obs space + masked ``R^{-1}``.

        Args:
            fc: Window forecast trajectory dataset (ensemble x time x system).
            obs_values: Observation values for the window, per obs time.
            obs_loc_indices: Per-obs-time observed system indices.
            obs_time_mask: Per-obs-time validity mask.
            obs_loc_mask: Per-obs-time / per-location validity mask.
            obs_window_indices: Model timestep nearest each obs time.

        Returns:
            ``(Xtraj, Yb, Y, rinv_diag)`` where ``Xtraj`` is the stacked
            forecast (ensemble x time x system), ``Yb`` the window-stacked
            obs-space ensemble (n_obs x ens), ``Y`` the flattened obs vector,
            and ``rinv_diag`` the masked diagonal of ``R^{-1}``.
        """
        # Per-obs-time observation operator Hs: (n_times, obs_dim, sys)
        if self.H is None:
            Hs = self._calc_default_H(obs_loc_indices)
        else:
            Hs = jnp.repeat(jnp.asarray(self.H)[jnp.newaxis],
                            obs_values.shape[0], axis=0)
        n_times, obs_dim = Hs.shape[0], Hs.shape[1]

        # Masked diagonal R^{-1} (obs-time mask x obs-location mask).
        sigma2 = jnp.atleast_1d(
            jnp.asarray(self.obs_error_sd, dtype=Hs.dtype) ** 2)
        # Broadcast per-obs (obs_dim,) OR scalar (1,) sigma2 across the n_times
        # slots so it matches the flattened (n_times*obs_dim,) mask vectors.
        sigma2_full = jnp.broadcast_to(sigma2, (n_times, obs_dim)).reshape(-1)
        time_mask = jnp.repeat(jnp.asarray(obs_time_mask, dtype=Hs.dtype),
                               obs_dim)
        loc_mask = jnp.asarray(obs_loc_mask, dtype=Hs.dtype).reshape(-1)
        rinv_diag = (time_mask * loc_mask) / sigma2_full

        # Gather obs-space predictions at the model step nearest each obs time.
        Xtraj = jnp.asarray(
                fc.to_stacked_array('system', ['ensemble', 'time']).data)
        owi = jnp.asarray(obs_window_indices)

        def _obs_pred(i):
            return Hs[i] @ Xtraj[:, owi[i], :].T          # (obs_dim, ens)

        Yb = jax.vmap(_obs_pred)(jnp.arange(n_times)).reshape(
                -1, self.ensemble_dim)
        Y = jnp.asarray(obs_values).reshape(-1)
        return Xtraj, Yb, Y, rinv_diag

    def _compute_analysis_4d(self,
                             Xb: ArrayLike,
                             Yb: ArrayLike,
                             Y: ArrayLike,
                             rinv_diag: ArrayLike,
                             rho: float = 1.0
                             ) -> ArrayLike:
        """ETKF transform from window-stacked obs, applied to ``Xb``.

        Thin wrapper composing :meth:`_compute_weights_4d` and
        :meth:`_apply_weights`; retained for the single-shot (window-start)
        analysis path and for direct callers/tests.
        """
        Wa, wa = self._compute_weights_4d(Yb, Y, rinv_diag, rho=rho)
        return self._apply_weights(Xb, Wa, wa)

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
        """Legacy single-shot analysis applied to the window-start ensemble.

        Unused by :meth:`_cycle_and_forecast_4d` (which selects the analysis
        time via ``analysis_time_index``); kept for API parity with the base
        cycler and for direct callers.
        """
        _, fc = self._step_forecast(xb0_ds, n_steps=self.steps_per_window)
        _, Yb, Y, rinv_diag = self._build_yb(
                fc, obs_values, obs_loc_indices, obs_time_mask, obs_loc_mask,
                obs_window_indices)

        Xb0 = jnp.asarray(
                xb0_ds.to_stacked_array('system', ['ensemble']).data).T
        Xa = self._compute_analysis_4d(
                Xb=Xb0, Yb=Yb, Y=Y, rinv_diag=rinv_diag,
                rho=self.multiplicative_inflation)

        return xb0_ds.assign(x=(['ensemble', 'i'], Xa.T))

    def _cycle_and_forecast_4d(self,
                               cur_state: xj.XjDataset,
                               filtered_idx: ArrayLike
                               ) -> tuple[xj.XjDataset, XarrayDatasetLike]:
        # 1. Get data; restore filtered_idx and obs-time mask (as in base).
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

        # 3. Time-invariant transform weights from the window innovations.
        Xtraj, Yb, Y, rinv_diag = self._build_yb(
                forecast_states, cur_obs_vals, cur_obs_loc_indices,
                obs_time_mask, cur_obs_loc_mask, obs_window_indices)
        Wa, wa = self._compute_weights_4d(
                Yb, Y, rinv_diag, rho=self.multiplicative_inflation)

        # 4. Apply the weights at the chosen in-window analysis time, then
        #    propagate the analysis ensemble forward to the window end so the
        #    next IC is every member forecast to the next-cycle start.
        tau = self._resolve_analysis_index()
        # Fresh per-cycle key (folded from the window index) enables structured
        # additive inflation; None when additive is off (byte-identical path).
        add_key = (jax.random.fold_in(
                    self._additive_key,
                    jnp.round(cur_time / self.analysis_window).astype(jnp.int32))
                   if self.additive_inflation > 0.0 else None)
        Xa = self._apply_weights(Xtraj[:, tau, :].T, Wa, wa,
                                 key=add_key)        # (system, ens)
        # Reuse the incoming state's variable/dim names so the analysis dataset
        # keeps the carry structure jax.lax.scan requires (ensemble-first).
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
            metrics = self._obs_metrics_4d(
                    cur_state, Xtraj, tau, Yb, Y, rinv_diag, Wa, wa,
                    cur_obs_vals, cur_obs_loc_indices, obs_time_mask,
                    cur_obs_loc_mask, obs_window_indices)
            return xj.from_xarray(next_state), (forecast_states, metrics)
        return xj.from_xarray(next_state), forecast_states

    def _obs_metrics_4d(self, cur_state, Xtraj, tau, Yb, Y, rinv_diag, Wa, wa,
                        cur_obs_vals, cur_obs_loc_indices, obs_time_mask,
                        cur_obs_loc_mask, obs_window_indices):
        """Causal (no-back-propagation) obs-space metrics for the 4D filters.

        O-F uses the window-stacked prior obs-space ensemble ``Yb`` (the same
        innovations that drive the transform).  O-A applies the transform
        weights at the ANALYSIS TIME ``tau`` (the SAME placement used for the
        IC handoff), re-forecasts the analysis ensemble from ``tau`` to the
        window end, and scores each observation at its own time WITHOUT
        acausal back-propagation: observations at times ``>= tau`` are scored
        against the analysis re-forecast to that obs time, while observations
        at times ``<= tau`` are scored against the analysis at ``tau`` (the
        earliest the analysis is valid).
        """
        dtype = Xtraj.dtype
        yb_bar = jnp.mean(Yb, axis=1)
        if self.H is None:
            Hs = self._calc_default_H(cur_obs_loc_indices)
        else:
            Hs = jnp.repeat(jnp.asarray(self.H)[jnp.newaxis],
                            cur_obs_vals.shape[0], axis=0)

        # Analysis-at-time closure (no additive key for a clean diagnostic);
        # the shared scorer selects tau vs window-start per oa_score_mode.
        def _analysis_at(t):
            return self._apply_weights(Xtraj[:, t, :].T, Wa, wa)

        ya_bar = self._score_oa_ya(
                cur_state, _analysis_at, tau, Hs, obs_window_indices, dtype)
        active = rinv_diag > 0
        sigma2_diag = jnp.where(
                active,
                1.0 / jnp.where(active, rinv_diag, jnp.ones_like(rinv_diag)),
                jnp.zeros_like(rinv_diag))
        ens_obs = Yb - yb_bar[:, None]

        # End-of-window O-A (next-cycle IC quality): analysis applied at tau,
        # propagated to the window END, scored against obs valid at the end.
        # The full end ENSEMBLE also gives the next-cycle IC spread.
        end_idx = self.steps_per_window - 1
        Xa_end_ens = self._xa_end_ensemble(
                cur_state, _analysis_at(tau), tau)          # (system, ens)
        Xa_end = jnp.mean(Xa_end_ens, axis=1)
        ya_end = jax.vmap(lambda i: Hs[i] @ Xa_end)(
                jnp.arange(Hs.shape[0])).reshape(-1).astype(dtype)
        # End-of-window analysis obs-space perturbations (obs x ens): drives
        # ``obs_space_spread_analysis_end`` (spread of the ensemble handed to
        # the next cycle -- NOT ``obs_space_spread_background`` in-window).
        Ya_end = jax.vmap(lambda i: Hs[i] @ Xa_end_ens)(
                jnp.arange(Hs.shape[0])).reshape(-1, self.ensemble_dim)
        ens_obs_end = (Ya_end - jnp.mean(Ya_end, axis=1)[:, None]).astype(dtype)
        owi = jnp.asarray(obs_window_indices)
        end_active = active & (jnp.repeat(owi, Hs.shape[1]) == end_idx)
        return dac_utils._obs_space_metrics(
                jnp.asarray(Y, dtype), yb_bar.astype(dtype),
                ya_bar, active, sigma2_diag, ens_obs=ens_obs,
                return_per_obs=(self._metrics_mode == "debug"), dtype=dtype,
                Hxa_end_mean=ya_end, end_active_mask=end_active,
                ens_obs_end=ens_obs_end)
