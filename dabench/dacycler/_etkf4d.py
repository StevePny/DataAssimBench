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
        self._analysis_time_spec = analysis_time_index
        super().__init__(*args, **kwargs)

    def _resolve_analysis_index(self) -> int:
        """Resolve ``analysis_time_index`` to an int in window bounds.

        ``self.steps_per_window`` is only known once :meth:`cycle` is running,
        so the spec is stored verbatim and resolved here (a static Python int,
        safe to use as a trajectory length under ``jax.lax.scan``).
        """
        n = int(self.steps_per_window)
        spec = self._analysis_time_spec
        if isinstance(spec, str):
            key = spec.lower()
            if key == "start":
                idx = 0
            elif key == "mid":
                idx = (n - 1) // 2
            elif key == "end":
                idx = n - 1
            else:
                raise ValueError(
                    "analysis_time_index string must be 'start', 'mid' or "
                    f"'end', got {spec!r}")
        else:
            idx = int(spec)
            if idx < 0:
                idx += n
        return max(0, min(idx, n - 1))

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

    def _compute_weights_4d(self,
                            Yb: ArrayLike,
                            Y: ArrayLike,
                            rinv_diag: ArrayLike,
                            rho: float = 1.0
                            ) -> tuple[ArrayLike, ArrayLike]:
        """Time-invariant ETKF transform weights from window-stacked obs.

        Args:
            Yb: Obs-space ensemble across the window, shape (n_obs, ens_dim).
            Y: Flattened observation vector, shape (n_obs,).
            rinv_diag: Diagonal of masked ``R^{-1}``, shape (n_obs,).
            rho: Multiplicative inflation factor (1.0 = none).

        Returns:
            ``(Wa, wa)`` -- the ensemble-perturbation transform matrix and the
            mean-update weight vector.
        """
        ensemble_dim = Yb.shape[1]
        U = jnp.ones((ensemble_dim, ensemble_dim)) / ensemble_dim
        I = jnp.identity(ensemble_dim)

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
        return Wa, wa

    def _apply_weights(self,
                       Xb: ArrayLike,
                       Wa: ArrayLike,
                       wa: ArrayLike,
                       key: ArrayLike | None = None
                       ) -> ArrayLike:
        """Apply transform weights to a background ensemble at one time.

        Args:
            Xb: Background ensemble, shape (system_dim, ens_dim).
            Wa: Ensemble-perturbation transform matrix, shape (ens, ens).
            wa: Mean-update weight vector, shape (ens,).
            key: Optional per-cycle PRNG key enabling structured additive
                inflation (:meth:`_apply_additive`); ``None`` skips it.

        Returns:
            Xa: Analysis ensemble, shape (system_dim, ens_dim).
        """
        ensemble_dim = Xb.shape[1]
        U = jnp.ones((ensemble_dim, ensemble_dim)) / ensemble_dim
        I = jnp.identity(ensemble_dim)

        Xb = Xb @ (I - U) + Xb @ U
        Xb_bar = jnp.mean(Xb, axis=1)
        Xb_pert = Xb @ (I - U)

        Xa_pert = Xb_pert @ Wa
        Xa_pert = self._apply_rtps(Xb_pert, Xa_pert)
        Xa_pert = self._apply_rtpp(Xb_pert, Xa_pert)
        if key is not None:
            Xa_pert = self._apply_additive(Xb_pert, Xa_pert, key)
        Xa_bar = Xb_bar + jnp.ravel(Xb_pert @ wa)
        v = jnp.ones((1, ensemble_dim))
        Xa = Xa_pert + Xa_bar[:, None] @ v
        return Xa

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

        return xj.from_xarray(next_state), forecast_states
