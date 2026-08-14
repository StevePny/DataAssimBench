"""Base class for Data Assimilation Cycler object (DACycler)"""

import numpy as np
import jax.numpy as jnp
import jax
import xarray as xr
from dabench import _xarray_jax as xj
from typing import Callable

import dabench.dacycler._utils as dac_utils
from dabench.model import Model


# For typing
ArrayLike = np.ndarray | jax.Array
XarrayDatasetLike = xr.Dataset | xj.XjDataset

class DACycler():
    """Base for all DACyclers

    Args:
        system_dim: System dimension
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
    # 3D-FGAT opt-in (First Guess at Appropriate Time).  When True on a 3D
    # (``_in_4d=False``) ensemble/var cycler, the analysis gathers observations
    # forward-looking (true obs times + per-obs window indices, exactly like the
    # 4D path) and computes innovations against the background trajectory AT each
    # obs time, while forming the analysis increment strictly at the single
    # in-window analysis time ``tau`` (textbook 3D-Var-FGAT).  Default False ->
    # the legacy single-slice 3D path (byte-identical numeric behaviour).
    _fgat: bool = False
    _return_metrics: bool = False       # set True only inside cycle()
    _metrics_mode: str = "default"      # "default" | "debug"
    # Baseline per-cycle scalar metrics emitted by EVERY cycler.  Cycler paths
    # may emit ADDITIONAL scalars (e.g. the FGAT tau-restricted comparison
    # pair); ``_assemble_metrics_ds`` stacks any extra scalar leaf present in
    # the emitted dict, so this tuple is only the guaranteed-present baseline.
    _METRIC_SCALARS = ("o_minus_f_rms", "o_minus_a_rms", "bias_f", "bias_a",
                       "obs_space_spread_background", "sigma_obs_max",
                       "n_active_obs", "o_minus_a_rms_end", "bias_a_end",
                       "n_active_obs_end", "obs_space_spread_analysis_end")
    # Per-obs (2-D ``(cycle, obs)``) debug leaves, emitted only in debug mode.
    _METRIC_PEROBS = ("o_minus_f", "o_minus_a", "obs_active")
    # After-run-accessible metrics container (set by cycle(); None otherwise).
    metrics = None
    # Warn once per run if debug-mode metrics exceed this many bytes (T42+
    # guard so per-obs arrays don't silently balloon host memory).
    _METRICS_WARN_BYTES = 256 * 1024 * 1024

    def __init__(self,
                 system_dim: int,
                 delta_t: float,
                 model_obj: Model,
                 B: ArrayLike | None = None,
                 R: ArrayLike | None = None,
                 H: ArrayLike | None = None,
                 h: Callable | None = None,
                 ):

        self.h = h
        self.H = H
        self.R = R
        self.B = B
        self.system_dim = system_dim
        self.delta_t = delta_t
        self.model_obj = model_obj


    def _calc_default_H(self,
                        obs_values: ArrayLike,
                        obs_loc_indices: ArrayLike
                        ) -> jax.Array:
        H = jnp.zeros((obs_values.flatten().shape[0], self.system_dim))
        H = H.at[jnp.arange(H.shape[0]), 
                 obs_loc_indices.flatten(),
                 ].set(1)
        return H

    def _calc_default_R(self,
                        obs_values: ArrayLike,
                        obs_error_sd: float
                        ) -> jax.Array:
        return jnp.identity(obs_values.flatten().shape[0])*(obs_error_sd**2)

    def _calc_default_B(self) -> jax.Array:
        """If B is not provided, identity matrix with shape (system_dim, system_dim."""
        return jnp.identity(self.system_dim)

    def _step_forecast(self,
                       xa: XarrayDatasetLike,
                       n_steps: int = 1
                       ) -> XarrayDatasetLike:
        """Perform forecast using model object"""
        return self.model_obj.forecast(xa, n_steps=n_steps)

    def _resolve_analysis_index(self) -> int:
        """Resolve ``analysis_time_index`` to an int in window bounds.

        ``self.steps_per_window`` is only known once :meth:`cycle` is running,
        so the spec (``self._analysis_time_spec``) is stored verbatim and
        resolved here (a static Python int, safe as a trajectory length under
        ``jax.lax.scan``).  Accepts ``"start"`` (0), ``"mid"`` (window centre),
        ``"end"`` (``steps_per_window - 1``), or an integer index (negative
        counts from the window end).  Shared by the FGAT (3D ensemble + Var3D)
        and 4D ensemble paths; subclasses that never set placement inherit the
        default ``"mid"`` via :attr:`_analysis_time_spec`.
        """
        n = int(self.steps_per_window)
        spec = getattr(self, "_analysis_time_spec", "mid")
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

    def _check_observability(self, all_filtered_idx, input_state) -> None:
        """One-shot observability / rank-deficiency check (subclass hook).

        No-op on the base cycler; :class:`ETKF` overrides it to warn once when
        the ensemble is too small to span the unstable subspace or when too few
        observed degrees of freedom are available to constrain the analysis.
        """
        return None

    def _step_cycle(self,
                    cur_state: XarrayDatasetLike,
                    obs_vals: ArrayLike,
                    obs_locs: ArrayLike,
                    obs_time_mask: ArrayLike,
                    obs_loc_mask: ArrayLike,
                    H: ArrayLike | None = None,
                    h: Callable | None =None,
                    R: ArrayLike | None = None,
                    B:ArrayLike | None = None,
                    **kwargs
                    ) -> XarrayDatasetLike:
        if H is not None or h is None:
            vals = self._cycle_obsop(
                    cur_state, obs_vals, obs_locs, obs_time_mask,
                    obs_loc_mask, H, R, B, **kwargs)
            return vals
        else:
            raise ValueError(
                'Only linear obs operators (H) are supported right now.')
            vals = self._cycle_general_obsop(
                    cur_state, obs_vals, obs_locs, obs_time_mask,
                    obs_loc_mask, h, R, B, **kwargs)
            return vals

    def _cycle_and_forecast(self,
                            cur_state: xj.XjDataset,
                            filtered_idx: ArrayLike
                            ) -> tuple[xj.XjDataset, XarrayDatasetLike]:
        # 1. Get data
        # 1-b. Calculate obs_time_mask and restore filtered_idx to original values
        cur_state = cur_state.to_xarray()
        cur_time = cur_state['_cur_time'].data
        cur_state = cur_state.drop_vars(['_cur_time'])
        obs_time_mask = filtered_idx > 0
        filtered_idx = filtered_idx - 1

        # 2. Calculate analysis
        cur_obs_vals = jnp.array(self._obs_vector[self._observed_vars].to_array().data).at[:, filtered_idx].get()
        cur_obs_loc_indices = jnp.array(self._obs_vector.system_index.data).at[:, filtered_idx].get()
        cur_obs_loc_mask = jnp.array(self._obs_loc_masks).at[:, filtered_idx].get().astype(bool)
        cur_obs_time_mask = jnp.repeat(obs_time_mask, cur_obs_vals.shape[-1])
        analysis = self._step_cycle(
                cur_state,
                cur_obs_vals,
                cur_obs_loc_indices,
                obs_loc_mask=cur_obs_loc_mask,
                obs_time_mask=cur_obs_time_mask
                )
        metrics = None
        if self._return_metrics:
            analysis, metrics = analysis      # _cycle_obsop returned a tuple
        # 3. Forecast next timestep
        next_state, forecast_states = self._step_forecast(analysis, n_steps=self.steps_per_window)
        next_state = next_state.assign(
            _cur_time = cur_time + self.analysis_window
            ).assign_coords(
                cur_state.coords).assign_attrs(cur_state.attrs)

        if self._return_metrics:
            return xj.from_xarray(next_state), (forecast_states, metrics)
        return xj.from_xarray(next_state), forecast_states

    def _cycle_and_forecast_4d(self,
                               cur_state: xj.XjDataset,
                               filtered_idx: ArrayLike
                               ) -> tuple[xj.XjDataset, XarrayDatasetLike]:
        # 1. Get data
        # 1-b. Calculate obs_time_mask and restore filtered_idx to original values
        cur_state = cur_state.to_xarray()
        cur_time = cur_state['_cur_time'].data
        cur_state = cur_state.drop_vars(['_cur_time'])
        obs_time_mask = filtered_idx > 0
        filtered_idx = filtered_idx - 1

        cur_obs_vals = jnp.array(self._obs_vector[self._observed_vars].to_stacked_array('system',['time']).data).at[filtered_idx].get()
        cur_obs_times = jnp.array(self._obs_vector.time.data).at[filtered_idx].get()
        cur_obs_loc_indices = jnp.array(self._obs_vector.system_index.data).at[:, filtered_idx].get().reshape(filtered_idx.shape[0], -1)
        cur_obs_loc_mask = jnp.array(self._obs_loc_masks).at[:, filtered_idx].get().astype(bool).reshape(filtered_idx.shape[0], -1)

        # Calculate obs window indices: closest model timesteps that match obs
        obs_window_indices =jnp.array([
                jnp.argmin(
                    jnp.abs(obs_time - (cur_time + self._model_timesteps))
                    ) for obs_time in cur_obs_times
            ])

        # 2. Calculate analysis
        analysis = self._step_cycle(
                cur_state,
                cur_obs_vals,
                cur_obs_loc_indices,
                obs_loc_mask=cur_obs_loc_mask,
                obs_time_mask=obs_time_mask,
                obs_window_indices=obs_window_indices
                )
        metrics = None
        if self._return_metrics:
            analysis, metrics = analysis      # _cycle_obsop returned a tuple

        # 3. Forecast forward
        next_state, forecast_states = self._step_forecast(analysis, n_steps=self.steps_per_window)
        next_state = next_state.assign(
            _cur_time = cur_time + self.analysis_window
            ).assign_coords(
                cur_state.coords).assign_attrs(cur_state.attrs)

        if self._return_metrics:
            return xj.from_xarray(next_state), (forecast_states, metrics)
        return xj.from_xarray(next_state), forecast_states

    def _cycle_and_forecast_fgat(self,
                                 cur_state: xj.XjDataset,
                                 filtered_idx: ArrayLike
                                 ) -> tuple[xj.XjDataset, XarrayDatasetLike]:
        """3D-FGAT scan step (overridden by FGAT-capable cyclers).

        The base class does not implement FGAT; a subclass that sets
        ``_fgat = True`` (e.g. :class:`ETKF`/:class:`LETKF`, :class:`Var3D`)
        must provide this.  Raising here makes an accidental ``_fgat`` on a
        non-FGAT cycler fail loudly rather than silently mis-cycle.
        """
        raise NotImplementedError(
            f"{type(self).__name__} enabled 3D-FGAT (_fgat=True) but does not "
            "implement _cycle_and_forecast_fgat.")

    def _prepare_cycle(self,
                       input_state: XarrayDatasetLike,
                       start_time: float | np.datetime64,
                       obs_vector: XarrayDatasetLike,
                       obs_error_sd: float | ArrayLike | None,
                       n_cycles: int,
                       analysis_window: float,
                       analysis_time_in_window: float | None
                       ) -> tuple[XarrayDatasetLike, ArrayLike]:
        """Populate obs/window attributes and build the padded obs indices.

        The shared, side-effecting pre-scan setup extracted verbatim from
        :meth:`cycle` (byte-identical): sets ``self._observed_vars``,
        ``self._data_vars``, ``self.analysis_window``, ``self.steps_per_window``,
        ``self._model_timesteps``, ``self._obs_vector``, ``self.obs_error_sd``,
        ``self._obs_loc_masks``; runs the one-shot observability check; and
        returns ``(input_state_with_time, all_filtered_padded)``.  Offline
        diagnostics reuse this so cycle 0 is reproduced with the SAME obs gather
        the scan would use.
        """
        # These could be different if observer doesn't observe all variables
        # For now, making them the same
        self._observed_vars = obs_vector['variable'].values
        self._data_vars = list(input_state.data_vars)

        if obs_error_sd is None:
            obs_error_sd = obs_vector.error_sd

        self.analysis_window = analysis_window

        # Whether this cycle rides a forward-looking window (obs gathered from
        # the analysis time forward across the window): the 4D path always does,
        # and a 3D-FGAT cycler does too (it needs the true obs times spanning
        # the window).  A plain 3D cycler keeps the legacy centered window.
        _forward_window = self._in_4d or self._fgat

        # If don't specify analysis_time_in_window, is assumed to be middle
        if analysis_time_in_window is None:
            if _forward_window:
                analysis_time_in_window = 0
            else:
                analysis_time_in_window = self.analysis_window/2

        # Steps per window + 1 to include start
        self.steps_per_window = round(analysis_window/self.delta_t) + 1
        self._model_timesteps = jnp.arange(self.steps_per_window)*self.delta_t

        # Time offset from middle of time window, for gathering observations
        _time_offset = (analysis_window/2) - analysis_time_in_window

        # Set up for jax.lax.scan, which is very fast
        all_times = dac_utils._get_all_times(
            start_time,
            analysis_window,
            n_cycles)


        if self.steps_per_window is None:
            self.steps_per_window = round(analysis_window/self.delta_t) + 1
        self._model_timesteps = jnp.arange(self.steps_per_window)*self.delta_t
        # Get the obs vectors for each analysis window
        all_filtered_idx = dac_utils._get_obs_indices(
            obs_times=jnp.array(obs_vector.time.values),
            analysis_times=all_times+_time_offset,
            start_inclusive=True,
            end_inclusive=_forward_window,
            analysis_window=analysis_window
        )
        input_state = input_state.assign(_cur_time=start_time)

        all_filtered_padded = dac_utils._pad_time_indices(all_filtered_idx, add_one=True)
        self._obs_vector=obs_vector
        self.obs_error_sd = obs_error_sd
        if obs_vector.stationary_observers:
            self._obs_loc_masks = jnp.ones(
                obs_vector[self._observed_vars].to_array().shape, dtype=bool)
        else:
            self._obs_loc_masks = ~np.isnan(
                obs_vector[self._observed_vars].to_array().data)
            self._obs_vector=self._obs_vector.fillna(0)

        # One-shot observability / rank-deficiency check (subclass hook; no-op
        # on the base).  Runs here (concrete host-side, before the scan, AFTER
        # self._obs_vector is set) so any warning fires exactly once per
        # cycle() rather than per scan step.
        self._check_observability(all_filtered_idx, input_state)
        return input_state, all_filtered_padded

    def cycle(self,
              input_state: XarrayDatasetLike,
              start_time: float | np.datetime64,
              obs_vector: XarrayDatasetLike,
              n_cycles: int,
              obs_error_sd: float | ArrayLike | None = None,
              analysis_window: float = 0.2,
              analysis_time_in_window: float | None = None,
              return_forecast: bool = False,
              return_metrics: bool = False,
              metrics_mode: str = "default"
              ) -> XarrayDatasetLike:
        """Perform DA cycle repeatedly, including analysis and forecast

        Args:
            input_state: Input state as a Xarray Dataset
            start_time: Starting time.
            obs_vector: Observations vector.
            n_cycles: Number of analysis cycles to run, each of length
                analysis_window.
            analysis_window: Time window from which to gather
                observations for DA Cycle.
            analysis_time_in_window: Where within analysis_window
                to perform analysis. For example, 0.0 is the start of the
                window. Default is None, which selects the middle of the
                window.
            return_forecast: If True, returns forecast at each model
                timestep. If False, returns only analyses, one per analysis
                cycle.
            return_metrics: If True, returns a tuple
                ``(analysis_ds, metrics_ds)`` where ``metrics_ds`` holds the
                native observation-space DA diagnostics (O-F/O-A RMS, biases,
                obs-space ensemble spread, sigma_obs_max, active-obs counts),
                one value per cycle. If False (default), returns EXACTLY the
                single analysis Dataset (byte-identical numeric path).
            metrics_mode: Granularity of the metrics (only consulted when
                ``return_metrics=True``). ``"default"`` emits per-cycle
                aggregate scalars; ``"debug"`` ALSO emits the full per-obs
                O-F/O-A arrays and an obs-active mask.
        """

        if metrics_mode not in ("default", "debug"):
            raise ValueError(
                f"metrics_mode must be 'default' or 'debug', "
                f"got {metrics_mode!r}")
        self._return_metrics = bool(return_metrics)
        self._metrics_mode = metrics_mode
        # Leak guard: drop any metrics retained from a PRIOR run at the START of
        # this cycle so an aborted/failed run never leaves a large array pinned,
        # and memory stays bounded to a single run's worth (the attribute is
        # replaced -- never appended -- on completion below).
        self.metrics = None

        # Shared setup: populate obs attributes, resolve the window, run the
        # one-shot observability check, and build the padded per-cycle obs
        # indices.  Factored into ``_prepare_cycle`` so offline diagnostics
        # (e.g. transform capture) can reproduce cycle 0 EXACTLY without
        # re-running the scan.
        input_state, all_filtered_padded = self._prepare_cycle(
            input_state, start_time, obs_vector, obs_error_sd, n_cycles,
            analysis_window, analysis_time_in_window)

        if self._in_4d:
            _fn = self._cycle_and_forecast_4d
        elif self._fgat:
            _fn = self._cycle_and_forecast_fgat
        else:
            _fn = self._cycle_and_forecast
        cur_state, scan_out = jax.lax.scan(
                _fn, xj.from_xarray(input_state), all_filtered_padded)
        if self._return_metrics:
            all_values, all_metrics = scan_out
        else:
            all_values = scan_out

        all_vals_ds = xr.Dataset(
            {var: (('cycle',) + tuple(all_values[var].dims),
                   all_values[var].data)
             for var in all_values.data_vars}
        ).rename_dims({'time': 'cycle_timestep'})

        analysis_ds = (all_vals_ds.drop_isel(cycle_timestep=-1)
                       if return_forecast
                       else all_vals_ds.isel(cycle_timestep=0))
        if not self._return_metrics:
            return analysis_ds
        metrics = self._assemble_metrics_ds(all_metrics)
        # Store on the instance for after-run access (see CyclerMetrics); also
        # return it (backward-compatible tuple contract).
        self.metrics = metrics
        return analysis_ds, metrics

    def clear_metrics(self) -> None:
        """Release the retained metrics container (frees host memory)."""
        self.metrics = None

    def _assemble_metrics_ds(self, all_metrics: dict) -> dac_utils.CyclerMetrics:
        """Stack scan-emitted per-cycle metrics leaves into a CyclerMetrics.

        ``jax.lax.scan`` stacks every metrics-pytree leaf along a leading
        ``cycle`` axis, so each scalar leaf is ``(n_cycles,)`` and each debug
        per-obs leaf is ``(n_cycles, obs_dim)``.  ANY scalar leaf present in the
        emitted dict (beyond the baseline ``_METRIC_SCALARS``) is stacked too,
        so cycler-specific extras (e.g. the FGAT tau-restricted comparison
        pair) propagate without every cycler having to emit them.
        """
        perobs = set(self._METRIC_PEROBS)
        data_vars = {}
        for k, v in all_metrics.items():
            if k in perobs:
                continue
            data_vars[k] = (('cycle',), np.asarray(v))
        if self._metrics_mode == "debug":
            for k in self._METRIC_PEROBS:
                if k in all_metrics:
                    data_vars[k] = (('cycle', 'obs'),
                                    np.asarray(all_metrics[k]))
        metrics = dac_utils.CyclerMetrics(xr.Dataset(data_vars))
        if metrics.nbytes > self._METRICS_WARN_BYTES:
            import warnings
            warnings.warn(
                f"Retained cycler metrics use {metrics.nbytes / 1024**2:.1f} "
                f"MiB ({metrics.memory_report()}); call clear_metrics() or use "
                f"metrics_mode='default' to reduce host memory.",
                stacklevel=2)
        return metrics
