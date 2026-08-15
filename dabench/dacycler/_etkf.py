"""Class for Ensemble Transform Kalman Filter (ETKF) DA Class"""

import warnings

import numpy as np
import jax
import jax.numpy as jnp
import xarray as xr
from dabench import _xarray_jax as xj
from typing import Callable

from dabench import dacycler
import dabench.dacycler._utils as dac_utils
from dabench.dacycler._utils import _resolve_eigh_impl, _solve_pa_wa
from dabench.model import Model


# For typing
ArrayLike = np.ndarray | jax.Array
XarrayDatasetLike = xr.Dataset | xj.XjDataset

class ETKF(dacycler.DACycler):
    """Ensemble transform Kalman filter DA Cycler

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
        ensemble_dim: Number of ensemble instances for ETKF. Default is
            4. Higher ensemble_dim increases accuracy but has performance cost.
        multiplicative_inflation: Scaling factor by which to multiply ensemble
            deviation. Default is 1.0 (no inflation).
        rtps_relaxation: Relaxation-to-Prior-Spread coefficient ``alpha``
            (Whitaker & Hamill 2012).  After the analysis the posterior
            perturbations are relaxed back toward the prior ensemble spread
            per coordinate:
            ``Xa_pert <- Xa_pert * (1 + alpha * (sigma_b - sigma_a) /
            sigma_a)``.  Default 0.0 (no relaxation); typical values 0.5-0.95.
            Applied independently of (and after) ``multiplicative_inflation``.
        additive_inflation: Absolute per-element RMS magnitude of structured
            additive perturbations injected into the analysis ensemble each
            cycle, setting an absolute spread floor (model-error
            representation).  The perturbations live in the forecast-difference
            subspace (random mean-zero recombination of the prior ensemble
            deviations), not white noise.  Default 0.0 (no injection); wired
            for the 4D path (:class:`ETKF4D`).  Applied after RTPS.
        additive_seed: Base PRNG seed for the additive injection; a fresh key
            is derived per cycle. Default 0.
        eigh_impl: SPD-solver backend for the ``K x K`` ETKF transform, shared
            with :class:`~dabench.dacycler.LETKF`.  ``None`` (default) uses
            ``jnp.linalg.eigh`` -- exact current behaviour.  ``"newton_schulz"``
            (alias ``"ns"``) selects the eigh-FREE, matmul-only Newton-Schulz
            inverse-square-root (batched GEMM on the accelerator, differentiable
            for ML-training integration); ``"qr"``/``"jacobi"`` select lax eigh
            variants.  All match to round-off for the SPD transform.
        ns_iters: Newton-Schulz coupled-iteration budget (only used when
            ``eigh_impl="newton_schulz"``).  Default 40 -- empirically the
            iteration count needed to reach the fp64 convergence floor across a
            wide conditioning range (kappa up to ~1e12) for a 128x128 SPD
            transform; 20 leaves high-kappa (>=1e10) transforms UNDER-converged
            even at fp64 (residual ~0.2-0.7).  The right budget is
            problem-dependent (system dimension, ensemble size, localization,
            inflation all shift the transform's conditioning): callers are
            strongly advised to run :func:`ns_iter_sweep` on a representative
            transform from their own system BEFORE trusting the NS path -- it
            reports, per (precision, conditioning), the per-step residual so you
            can pick ``ns_iters`` and confirm precision suffices (fp32 has a hard
            precision floor set by conditioning that NO ``ns_iters`` overcomes).
        ns_resid_warn: Threshold on the Newton-Schulz relative convergence
            residual ``||Z A Z - I||_F / sqrt(K)`` above which a warning is
            emitted suggesting more ``ns_iters`` (fp64-calibrated; loosen under
            fp32).  Default 1e-6.  Only checked when the analysis is evaluated
            eagerly (e.g. the ``--da-forensics`` path); silently skipped inside
            ``jax.lax.scan`` to avoid tracer leaks.
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
                 ensemble_dim: int = 4,
                 multiplicative_inflation: float = 1.0,
                 rtps_relaxation: float = 0.0,
                 rtpp_relaxation: float = 0.0,
                 additive_inflation: float = 0.0,
                 additive_seed: int = 0,
                 eigh_impl: str | None = None,
                 ns_iters: int = 40,
                 ns_resid_warn: float = 1e-6,
                 fgat: bool = False,
                 analysis_time_index: int | str = "mid",
                 fgat_representativeness: bool = True,
                 oa_score_mode: str = "causal",
                 ):

        self.ensemble_dim = ensemble_dim
        self.multiplicative_inflation = multiplicative_inflation
        # 3D-FGAT opt-in.  When True the 3D cycler assimilates observations
        # distributed across the window: innovations are formed against the
        # background trajectory AT each obs time, but the analysis increment is
        # built strictly at the single in-window analysis time ``tau``
        # (textbook 3D-Var-FGAT single-time covariance).  Default False keeps
        # the legacy single-slice 3D path (byte-identical).  ``_fgat`` is a
        # class-level flag consulted by the base cycler; set the instance value
        # only when FGAT is requested on a 3D (non-4D) cycler.
        if fgat and not self._in_4d:
            self._fgat = True
        # In-window analysis-time placement for FGAT / 4D (resolved to a model
        # step once ``steps_per_window`` is known); stored verbatim.
        self._analysis_time_spec = analysis_time_index
        # Whether to inflate R by the number of distinct obs times per window to
        # account for representativeness error when >1 obs time is mapped onto
        # the single analysis-time covariance (FGAT only).
        self.fgat_representativeness = bool(fgat_representativeness)
        # Observation-minus-analysis (O-A) scoring convention for the obs-space
        # metrics (does NOT affect the analysis itself):
        #   "causal" (default) -- no acausal back-propagation: obs at times
        #       >= tau are scored against the analysis re-forecast forward to
        #       the obs time; obs at times <= tau are scored against the
        #       analysis at tau (the earliest it is valid).
        #   "time_matched" -- each obs is scored against the analysis valid at
        #       that obs's own time by placing a DIAGNOSTIC analysis at the
        #       window start and re-forecasting it across the window, so every
        #       obs time is reachable forward.  Restores the O-A <= O-F sanity
        #       property; intended for validation/diagnostics.
        if oa_score_mode not in ("causal", "time_matched"):
            raise ValueError(
                "oa_score_mode must be 'causal' or 'time_matched', got "
                f"{oa_score_mode!r}")
        self.oa_score_mode = oa_score_mode
        self.rtps_relaxation = float(rtps_relaxation)
        self.rtpp_relaxation = float(rtpp_relaxation)
        self.additive_inflation = float(additive_inflation)
        self._additive_key = jax.random.PRNGKey(int(additive_seed))
        # SPD-solver backend for the K x K transform (shared with LETKF via
        # ``_solve_pa_wa``).  ``None`` -> eigh (exact current behaviour);
        # ``"newton_schulz"`` -> eigh-free matmul-only inverse-sqrt.
        self.eigh_impl = _resolve_eigh_impl(eigh_impl)
        self.ns_iters = int(ns_iters)
        self.ns_resid_warn = float(ns_resid_warn)
        # Per-cycle Newton-Schulz convergence residual (updated each analysis;
        # concrete only when the analysis runs eagerly, else stays None).
        self.ns_resid = None

        super().__init__(system_dim=system_dim,
                         delta_t=delta_t,
                         model_obj=model_obj,
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

    def _check_observability(self, all_filtered_idx, input_state) -> None:
        """Warn once when the ensemble / obs network is likely rank-deficient.

        Two heuristics, evaluated once per :meth:`~DACycler.cycle` (host-side,
        before the scan, so each warning fires at most once):

        * **Ensemble rank.**  The ensemble spans an at-most ``(K-1)``-dim
          subspace.  If ``K-1 < system_dim`` the analysis cannot correct the
          full state; for a chaotic system ``K-1`` must at least cover the
          unstable-and-neutral subspace (dimension unknown a priori, so
          ``system_dim`` is used as a conservative proxy).  This is a heuristic,
          not an error -- localization often makes a smaller ensemble viable --
          so it warns rather than raises.
        * **Observed DOF.**  If the smallest number of observed values (obs
          times x observed locations) in any non-empty cycle is below
          ``system_dim``, the obs network may under-constrain the analysis for
          those cycles.

        ``all_filtered_idx`` is the pre-scan list of per-cycle obs-index arrays
        (concrete numpy/JAX arrays), so obs counts are known here without
        entering the traced scan.
        """
        K = int(self.ensemble_dim)
        sys = int(self.system_dim)
        if K - 1 < sys:
            warnings.warn(
                f"ensemble_dim={K} gives an ensemble subspace of rank <= "
                f"{K - 1}, which is smaller than system_dim={sys}; the "
                "analysis cannot correct the full state and may be "
                "rank-deficient (increase ensemble_dim so K-1 >= the "
                "unstable-subspace dimension, or rely on localization).",
                stacklevel=2)
        # Observed values per cycle = (n obs times) x (observed locations per
        # time).  Observed locations per obs time = the last axis of the obs
        # vector's per-time observed-value array; fall back to 1 if unavailable.
        loc_per_time = 1
        try:
            si = np.asarray(self._obs_vector.system_index.data)
            # system_index is (..., n_times, obs_dim) for stationary observers
            # or (n_times, obs_dim); the observed-locations-per-time is the
            # trailing axis when it is at least 2-D.
            if si.ndim >= 2:
                loc_per_time = int(si.shape[-1])
        except (AttributeError, ValueError, IndexError):
            loc_per_time = 1
        try:
            counts = [int(np.asarray(idx).shape[0]) * loc_per_time
                      for idx in all_filtered_idx]
        except (TypeError, ValueError):
            counts = []
        nonempty = [c for c in counts if c > 0]
        if nonempty:
            min_obs = min(nonempty)
            if min_obs < sys:
                warnings.warn(
                    f"at least one analysis cycle has only {min_obs} observed "
                    f"value(s), fewer than system_dim={sys}; those cycles may "
                    "be observation-rank-deficient (the analysis is "
                    "under-constrained).",
                    stacklevel=2)

    def _calc_default_H_4d(self,
                           obs_loc_indices: ArrayLike
                           ) -> jax.Array:
        """Per-obs-time selection operator, shape (n_times, obs_dim, sys).

        The FGAT/window analog of :meth:`DACycler._calc_default_H`: one linear
        selector per obs time built from that time's observed system indices.
        """
        Hs = jnp.zeros((obs_loc_indices.shape[0], obs_loc_indices.shape[1],
                        self.system_dim))
        for i in range(Hs.shape[0]):
            Hs = Hs.at[i, jnp.arange(Hs.shape[1]), obs_loc_indices[i]].set(1.0)
        return Hs

    def _build_fgat_yb(self,
                       fc: XarrayDatasetLike,
                       tau: int,
                       obs_values: ArrayLike,
                       obs_loc_indices: ArrayLike,
                       obs_time_mask: ArrayLike,
                       obs_loc_mask: ArrayLike,
                       obs_window_indices: ArrayLike,
                       ) -> tuple[ArrayLike, ArrayLike, ArrayLike, ArrayLike,
                                  ArrayLike]:
        """Strict 3D-FGAT obs-space assembly (single-time covariance).

        The obs-space ensemble ``Yb`` is built from the background
        perturbations at the SINGLE analysis time ``tau`` (single-time
        covariance, i.e. textbook 3D-Var-FGAT), while the innovation mean uses
        the background ensemble MEAN at each observation's true window time
        (First Guess at Appropriate Time).  An "effective" observation vector
        ``Y_eff = d + yb_bar_tau`` is returned so the existing ETKF transform
        (which forms its innovation as ``Y - mean(Yb)``) reproduces exactly the
        FGAT innovation ``d`` when handed ``(Yb, Y_eff)``.

        Args:
            fc: Window forecast trajectory dataset (ensemble x time x system).
            tau: In-window analysis-time model step index.
            obs_values: Per-obs-time observation values.
            obs_loc_indices: Per-obs-time observed system indices.
            obs_time_mask: Per-obs-time validity mask.
            obs_loc_mask: Per-obs-time / per-location validity mask.
            obs_window_indices: Model timestep nearest each obs time.

        Returns:
            ``(Xtraj, Yb, Y_eff, rinv_diag, n_obs_times)`` where ``Xtraj`` is
            the stacked forecast (ensemble x time x system), ``Yb`` the
            tau-time obs-space ensemble (n_obs x ens), ``Y_eff`` the effective
            obs vector, ``rinv_diag`` the masked diagonal of ``R^{-1}`` (with
            representativeness inflation folded in), and ``n_obs_times`` the
            number of DISTINCT active obs times in the window (a scalar).
        """
        if self.H is None:
            Hs = self._calc_default_H_4d(obs_loc_indices)
        else:
            Hs = jnp.repeat(jnp.asarray(self.H)[jnp.newaxis],
                            obs_values.shape[0], axis=0)
        n_times, obs_dim = Hs.shape[0], Hs.shape[1]

        # Masked diagonal R^{-1} (obs-time mask x obs-location mask).
        sigma2 = jnp.atleast_1d(
            jnp.asarray(self.obs_error_sd, dtype=Hs.dtype) ** 2)
        sigma2_full = jnp.broadcast_to(sigma2, (n_times, obs_dim)).reshape(-1)
        time_mask = jnp.repeat(jnp.asarray(obs_time_mask, dtype=Hs.dtype),
                               obs_dim)
        loc_mask = jnp.asarray(obs_loc_mask, dtype=Hs.dtype).reshape(-1)

        # Representativeness inflation: when the window holds more than one
        # distinct active obs time, mapping temporally-offset obs onto the
        # single tau-time covariance introduces representativeness error.
        # Inflate R by the number of distinct active obs times so each obs is
        # de-weighted accordingly (a diagonal, per-obs scaling).
        n_obs_times = jnp.sum(jnp.asarray(obs_time_mask, dtype=Hs.dtype))
        if self.fgat_representativeness:
            repr_factor = jnp.maximum(n_obs_times, jnp.ones_like(n_obs_times))
        else:
            repr_factor = jnp.ones_like(n_obs_times)
        rinv_diag = (time_mask * loc_mask) / (sigma2_full * repr_factor)

        Xtraj = jnp.asarray(
                fc.to_stacked_array('system', ['ensemble', 'time']).data)
        owi = jnp.asarray(obs_window_indices)

        # Perturbations at tau (single-time covariance): H_i @ X(tau).
        Xtau = Xtraj[:, tau, :]                            # (ens, system)
        xbar_tau = jnp.mean(Xtau, axis=0)                  # (system,)

        def _yb_tau(i):
            return Hs[i] @ Xtau.T                          # (obs_dim, ens)

        Yb = jax.vmap(_yb_tau)(jnp.arange(n_times)).reshape(
                -1, self.ensemble_dim)
        yb_bar_tau = jnp.mean(Yb, axis=1)                  # (n_obs,)

        # Innovation mean at each obs's true time: d = y - H_i @ xbar(t_obs_i).
        def _hxbar_tobs(i):
            xbar_i = jnp.mean(Xtraj[:, owi[i], :], axis=0)  # (system,)
            return Hs[i] @ xbar_i                           # (obs_dim,)

        Hxbar_tobs = jax.vmap(_hxbar_tobs)(jnp.arange(n_times)).reshape(-1)
        Y = jnp.asarray(obs_values).reshape(-1)
        d = Y - Hxbar_tobs                                  # FGAT innovation

        # Effective obs vector so the transform's (Y_eff - yb_bar_tau) == d.
        Y_eff = d + yb_bar_tau
        return Xtraj, Yb, Y_eff, rinv_diag, n_obs_times

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

    def _apply_rtps(self,
                    Xb_pert: ArrayLike,
                    Xa_pert: ArrayLike
                    ) -> ArrayLike:
        """Relaxation to Prior Spread (Whitaker & Hamill 2012).

        Relaxes the analysis perturbations toward the prior ensemble
        standard deviation per coordinate, countering the systematic
        spread reduction of the analysis step:

            ``Xa_pert <- Xa_pert * (1 + alpha * (sigma_b - sigma_a) / sigma_a)``

        where ``sigma_b`` / ``sigma_a`` are the per-coordinate prior /
        posterior ensemble standard deviations (the ``(K-1)`` factor
        cancels in the ratio).  ``alpha = rtps_relaxation``; ``alpha = 0``
        is an exact no-op, so this is always safe to call.

        Args:
            Xb_pert: Prior perturbations, shape ``(system_dim, ens)``.
            Xa_pert: Posterior perturbations, shape ``(system_dim, ens)``.

        Returns:
            The relaxed analysis perturbations, same shape as ``Xa_pert``.
        """
        alpha = self.rtps_relaxation
        sigma_b = jnp.std(Xb_pert, axis=1)
        sigma_a = jnp.std(Xa_pert, axis=1)
        scale = 1.0 + alpha * (sigma_b - sigma_a) / jnp.where(
                sigma_a > 0, sigma_a, 1.0)
        return Xa_pert * scale[:, None]

    def _apply_rtpp(self,
                    Xb_pert: ArrayLike,
                    Xa_pert: ArrayLike
                    ) -> ArrayLike:
        """Relaxation to Prior Perturbations (Zhang et al. 2004).

        Relaxes the analysis perturbations toward the PRIOR perturbations
        themselves (not just their spread), a linear blend per member:

            ``Xa_pert <- (1 - alpha) * Xa_pert + alpha * Xb_pert``

        where ``alpha = rtpp_relaxation``.  Unlike RTPS (which rescales the
        posterior spread per coordinate), RTPP re-injects the prior
        perturbation STRUCTURE, so when the prior ensemble is expanding
        along the flow's growing modes it carries that structure into the
        analysis and resists collapse over cycles.  ``alpha = 0`` is an
        exact no-op, so this is always safe to call.

        Args:
            Xb_pert: Prior perturbations, shape ``(system_dim, ens)``.
            Xa_pert: Posterior perturbations, shape ``(system_dim, ens)``.

        Returns:
            The relaxed analysis perturbations, same shape as ``Xa_pert``.
        """
        alpha = self.rtpp_relaxation
        return (1.0 - alpha) * Xa_pert + alpha * Xb_pert

    def _apply_additive(self,
                        Xb_pert: ArrayLike,
                        Xa_pert: ArrayLike,
                        key: ArrayLike
                        ) -> ArrayLike:
        """Structured additive inflation of the analysis perturbations.

        Injects mean-zero perturbations in the forecast-difference subspace (a
        random recombination of the prior ensemble deviations ``Xb_pert``,
        NOT white noise), rescaled to an absolute per-element RMS of
        ``additive_inflation``, and adds them to the posterior perturbations.
        This sets an absolute spread floor each cycle (model-error
        representation), countering the systematic analysis contraction that
        multiplicative levers (inflation / RTPS) cannot arrest when the prior
        itself is collapsing.  ``additive_inflation = 0`` is an exact no-op.

        Args:
            Xb_pert: Prior perturbations, shape ``(system_dim, ens)``.
            Xa_pert: Posterior perturbations, shape ``(system_dim, ens)``.
            key: PRNG key for this cycle's injection.

        Returns:
            The inflated analysis perturbations, same shape as ``Xa_pert``.
        """
        sigma = self.additive_inflation
        if sigma <= 0.0:
            return Xa_pert
        ensemble_dim = Xa_pert.shape[1]
        # dtype-relative Z: jax.random.normal defaults to float64 when
        # jax_enable_x64 is on (dabench force-enables it), which would promote a
        # float32 analysis to float64 (E = Xb_pert @ Z) and break the fp32
        # lax.scan carry.  Pin Z to the perturbation dtype (the DTYPE INVARIANT:
        # derive every intermediate from the input array dtype, never force one).
        Z = jax.random.normal(key, (ensemble_dim, ensemble_dim),
                              dtype=Xa_pert.dtype)
        E = Xb_pert @ Z
        E = E - jnp.mean(E, axis=1, keepdims=True)   # mean-zero across members
        rms = jnp.sqrt(jnp.mean(E ** 2))
        E = E * (sigma / jnp.where(rms > 0, rms, 1.0))
        return Xa_pert + E

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
          Y: Observation array with shape (obs_time_dim, observation_dim)
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
            # SPD transform ``A = (K-1)/rho I + Yb_pert^T R^{-1} Yb_pert``; the
            # shared solver returns ``Pa = A^{-1}`` and ``Wa=((K-1)A^{-1})^½``
            # via eigh (``None``/``qr``/``jacobi``) or the eigh-free Newton-
            # Schulz path (``self.eigh_impl``).  Both avoid ``sqrtm``/``schur``
            # (no CUDA lowering).
            A = (ensemble_dim-1)/rho*I + Yb_pert.T @ Rinv @ Yb_pert
            Pa_ens, Wa, ns_resid = _solve_pa_wa(
                A, self.eigh_impl, self.ns_iters)
            self._update_ns_diagnostics(ns_resid)
        else:
            Rinv = jnp.zeros_like(R, dtype=R.dtype)
            Pa_ens = jnp.zeros((ensemble_dim, ensemble_dim), dtype=R.dtype)
            Wa = jnp.zeros((ensemble_dim, ensemble_dim), dtype=R.dtype)

        wa = Pa_ens @ Yb_pert.T @ Rinv @ (Y.flatten()-yb_bar)

        Xa_pert = Xb_pert @ Wa
        Xa_pert = self._apply_rtps(Xb_pert, Xa_pert)
        Xa_pert = self._apply_rtpp(Xb_pert, Xa_pert)

        Xa_bar = Xb_bar + jnp.ravel(Xb_pert @ wa)

        v = jnp.ones((1, ensemble_dim))
        Xa = Xa_pert + Xa_bar[:, None] @ v

        return Xa

    def _compute_weights_4d(self,
                            Yb: ArrayLike,
                            Y: ArrayLike,
                            rinv_diag: ArrayLike,
                            rho: float = 1.0
                            ) -> tuple[ArrayLike, ArrayLike]:
        """ETKF transform weights from an obs-space ensemble + diagonal R^{-1}.

        Shared by the window-stacked 4D path (:class:`ETKF4D`) and the strict
        3D-FGAT path (which passes the tau-time ``Yb`` and an effective ``Y``
        so ``Y - mean(Yb)`` equals the FGAT innovation).

        Args:
            Yb: Obs-space ensemble, shape (n_obs, ens_dim).
            Y: Flattened (effective) observation vector, shape (n_obs,).
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
        # SPD transform ``A = (K-1)/rho I + Yb_pert^T R^{-1} Yb_pert``; the
        # shared solver returns ``Pa = A^{-1}`` and ``Wa = ((K-1)A^{-1})^½``
        # via eigh (``None``/``qr``/``jacobi``) or the eigh-free Newton-Schulz
        # path (``self.eigh_impl``).  Both avoid ``sqrtm``/``schur`` (no CUDA
        # lowering).
        A = (ensemble_dim - 1) / rho * I + YtRinv @ Yb_pert
        Pa_ens, Wa, ns_resid = _solve_pa_wa(A, self.eigh_impl, self.ns_iters)
        self._update_ns_diagnostics(ns_resid)
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

    def _score_oa_ya(self,
                     cur_state: XarrayDatasetLike,
                     analysis_at,
                     tau: int,
                     Hs: ArrayLike,
                     obs_window_indices: ArrayLike,
                     dtype) -> ArrayLike:
        """Obs-space analysis mean ``ya_bar`` under the O-A scoring convention.

        Shared by the 3D-FGAT and 4D obs-space metrics.  ``analysis_at(t)`` is
        a callable returning the analysis ensemble ``(system, ens)`` produced
        by applying the transform at in-window time index ``t`` (the global or
        localized kernel is injected by the caller).  The scoring convention is
        governed by :attr:`oa_score_mode`:

        * ``"causal"`` -- analysis placed at ``tau``, re-forecast from ``tau``
          to the window end; each obs at window index ``k`` is scored at
          ``clip(k - tau, 0, ..)`` (obs before ``tau`` clamp to the tau-time
          analysis; no back-propagation).
        * ``"time_matched"`` -- analysis placed at the WINDOW START, re-forecast
          across the whole window; each obs is scored at its own window index
          ``k`` (a like-for-like forward score, restoring O-A <= O-F).
        """
        xdims = cur_state['x'].dims
        owi = jnp.asarray(obs_window_indices)
        n_times = Hs.shape[0]
        if self.oa_score_mode == "time_matched":
            Xa0 = analysis_at(0)                      # (system, ens)
            Xa0_o = Xa0.T if xdims[0] == 'ensemble' else Xa0
            ana0_ds = cur_state.assign(x=(xdims, Xa0_o))
            _, ana_fc = self._step_forecast(
                    ana0_ds, n_steps=self.steps_per_window)
            Xa_traj = jnp.asarray(
                    ana_fc.to_stacked_array('system', ['ensemble', 'time']).data)
            offset = 0
        else:
            Xa = analysis_at(tau)                     # (system, ens)
            Xa_o = Xa.T if xdims[0] == 'ensemble' else Xa
            ana_tau_ds = cur_state.assign(x=(xdims, Xa_o))
            _, ana_fc = self._step_forecast(
                    ana_tau_ds, n_steps=self.steps_per_window - tau)
            Xa_traj = jnp.asarray(
                    ana_fc.to_stacked_array('system', ['ensemble', 'time']).data)
            offset = tau

        def _ya(i):
            j = jnp.clip(owi[i] - offset, 0, Xa_traj.shape[1] - 1)
            xbar_i = jnp.mean(Xa_traj[:, j, :], axis=0)
            return Hs[i] @ xbar_i

        return jax.vmap(_ya)(jnp.arange(n_times)).reshape(-1).astype(dtype)

    def _fgat_analysis(self,
                       Xb_tau: ArrayLike,
                       Yb: ArrayLike,
                       Y_eff: ArrayLike,
                       rinv_diag: ArrayLike,
                       obs_loc_flat: ArrayLike,
                       rho: float,
                       key: ArrayLike | None = None,
                       cycle_idx=None,
                       obs_latlon_t=None) -> ArrayLike:
        """Strict 3D-FGAT analysis at the single analysis time ``tau``.

        Global (non-localized) ETKF transform: builds the weights from the
        tau-time obs-space ensemble ``Yb`` + effective obs vector ``Y_eff``
        (whose innovation ``Y_eff - mean(Yb)`` is the FGAT innovation) and
        applies them to the tau-time background ensemble ``Xb_tau``.
        :class:`~dabench.dacycler.LETKF` overrides this with the localized
        per-gridpoint solve.  ``obs_loc_flat`` is accepted for signature parity
        with the localized override (unused here).

        Args:
            Xb_tau: Background ensemble at tau, shape (system_dim, ens).
            Yb: tau-time obs-space ensemble, shape (n_obs, ens).
            Y_eff: Effective obs vector, shape (n_obs,).
            rinv_diag: Masked diagonal ``R^{-1}``, shape (n_obs,).
            obs_loc_flat: Flattened observed grid indices (localized override).
            rho: Multiplicative inflation factor.
            key: Optional per-cycle additive-inflation PRNG key.
            cycle_idx: Accepted for signature parity with the localized
                Regime-B override (unused in the global ETKF path).
            obs_latlon_t: Accepted for signature parity with the localized
                Regime-B callback override (unused in the global ETKF path).

        Returns:
            Analysis ensemble at tau, shape (system_dim, ens).
        """
        del obs_loc_flat, cycle_idx, obs_latlon_t
        Wa, wa = self._compute_weights_4d(Yb, Y_eff, rinv_diag, rho=rho)
        return self._apply_weights(Xb_tau, Wa, wa, key=key)

    def _cycle_and_forecast_fgat(self,
                                 cur_state: xj.XjDataset,
                                 filtered_idx: ArrayLike
                                 ) -> tuple[xj.XjDataset, XarrayDatasetLike]:
        """3D-FGAT scan step for the ensemble filters (ETKF / LETKF).

        Mirrors :meth:`ETKF4D._cycle_and_forecast_4d` for the obs assembly and
        the tau-placement re-forecast handoff, but the analysis increment is
        formed strictly at the single analysis time ``tau`` from the tau-time
        covariance (:meth:`_build_fgat_yb` / :meth:`_fgat_analysis`).
        """
        # 1. Get data; restore filtered_idx and obs-time mask (as in 4D).
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

        # 2. Prior forecast of the incoming ensemble across the window; scored
        #    trajectory for this cycle and the source of the FGAT innovations.
        _, forecast_states = self._step_forecast(
                cur_state, n_steps=self.steps_per_window)

        # 3. Strict-FGAT obs-space assembly: perturbations at tau, innovation
        #    mean at each obs's true window time.
        tau = self._resolve_analysis_index()
        Xtraj, Yb, Y_eff, rinv_diag, _ = self._build_fgat_yb(
                forecast_states, tau, cur_obs_vals, cur_obs_loc_indices,
                obs_time_mask, cur_obs_loc_mask, obs_window_indices)
        obs_loc_flat = jnp.asarray(cur_obs_loc_indices).reshape(-1)

        # 4. Analysis at tau, then re-forecast tau -> window end for next IC.
        add_key = (jax.random.fold_in(
                    self._additive_key,
                    jnp.round(cur_time / self.analysis_window).astype(jnp.int32))
                   if self.additive_inflation > 0.0 else None)
        Xb_tau = Xtraj[:, tau, :].T                            # (system, ens)
        # Regime-B moving-obs geometry (LETKF-FGAT only; None otherwise):
        # precomputed-stack row (cyc) or live host-callback obs positions.
        cyc = getattr(self, "_cycle_index", lambda _t: None)(cur_time)
        obs_ll_t = getattr(self, "_callback_obs_latlon_4d",
                           lambda _i: None)(cur_obs_loc_indices)
        Xa = self._fgat_analysis(
                Xb_tau, Yb, Y_eff, rinv_diag, obs_loc_flat,
                rho=self.multiplicative_inflation, key=add_key, cycle_idx=cyc,
                obs_latlon_t=obs_ll_t)

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
            metrics = self._obs_metrics_fgat(
                    cur_state, Xtraj, tau, Yb, Y_eff, rinv_diag, obs_loc_flat,
                    cur_obs_vals, cur_obs_loc_indices, obs_time_mask,
                    cur_obs_loc_mask, obs_window_indices, cycle_idx=cyc,
                    obs_latlon_t=obs_ll_t)
            return xj.from_xarray(next_state), (forecast_states, metrics)
        return xj.from_xarray(next_state), forecast_states

    def _obs_metrics_fgat(self, cur_state, Xtraj, tau, Yb, Y_eff, rinv_diag,
                          obs_loc_flat, cur_obs_vals, cur_obs_loc_indices,
                          obs_time_mask, cur_obs_loc_mask, obs_window_indices,
                          cycle_idx=None, obs_latlon_t=None):
        """Obs-space metrics for the 3D-FGAT ensemble path.

        O-F is the FGAT innovation ``d = y - H_i xbar(t_obs_i)`` scored at each
        obs's APPROPRIATE time (First Guess at Appropriate Time): the O-F first
        guess is the background ensemble mean at that obs's true window time,
        NOT the single analysis time ``tau``.  O-A is scored at the single
        ANALYSIS time ``tau`` -- ``O-A_i = y_i - H_i xbar^a(tau)`` -- for every
        in-window obs, since the FGAT increment is formed at ``tau`` from the
        ``tau``-time covariance and the analysis is only strictly valid there
        (no re-forecast, no per-obs time matching; the ``oa_score_mode`` flag
        governs only the 4D path).  The obs-space ensemble spread is the
        single-time (``tau``) covariance the transform actually uses.
        """
        dtype = Xtraj.dtype
        yb_bar = jnp.mean(Yb, axis=1)
        Y = jnp.asarray(cur_obs_vals, dtype).reshape(-1)

        if self.H is None:
            Hs = self._calc_default_H_4d(cur_obs_loc_indices)
        else:
            Hs = jnp.repeat(jnp.asarray(self.H)[jnp.newaxis],
                            cur_obs_vals.shape[0], axis=0)

        # O-F first guess at each obs's APPROPRIATE time (FGAT): the background
        # ensemble mean projected at that obs's true window step -- identical to
        # the first guess used to form the analysis innovation in
        # :meth:`_build_fgat_yb`.  (Scoring O-F at ``tau`` instead would defeat
        # the purpose of FGAT.)
        owi = jnp.asarray(obs_window_indices)

        def _hxbar_tobs(i):
            xbar_i = jnp.mean(Xtraj[:, owi[i], :], axis=0)
            return Hs[i] @ xbar_i

        yf_bar = jax.vmap(_hxbar_tobs)(
                jnp.arange(Hs.shape[0])).reshape(-1).astype(dtype)

        # O-A scored at the single analysis time ``tau`` (no re-forecast, no
        # per-obs time matching): the FGAT increment is formed at ``tau`` from
        # the ``tau``-time covariance, so the analysis is only strictly valid
        # there.  Score every in-window obs against the analysis mean projected
        # at ``tau``: O-A_i = y_i - H_i xbar^a(tau).  (No additive key -- a
        # clean deterministic diagnostic.)
        Xa_tau = self._fgat_analysis(
                Xtraj[:, tau, :].T, Yb, Y_eff, rinv_diag, obs_loc_flat,
                rho=self.multiplicative_inflation, key=None, cycle_idx=cycle_idx,
                obs_latlon_t=obs_latlon_t)
        xbar_a_tau = jnp.mean(Xa_tau, axis=1)                 # (system,)
        ya_bar = jax.vmap(lambda i: Hs[i] @ xbar_a_tau)(
                jnp.arange(Hs.shape[0])).reshape(-1).astype(dtype)
        active = rinv_diag > 0
        sigma2_diag = jnp.where(
                active, 1.0 / jnp.where(active, rinv_diag,
                                        jnp.ones_like(rinv_diag)),
                jnp.zeros_like(rinv_diag))
        ens_obs = Yb - yb_bar[:, None]

        # End-of-window O-A (next-cycle IC quality): score obs valid at the
        # window END against the analysis mean propagated to the window end
        # (the state that seeds the next cycle).  End-obs = owi == last index.
        # The full end ENSEMBLE also gives the next-cycle IC spread.
        end_idx = self.steps_per_window - 1
        Xa_end = self._xa_end_ensemble(cur_state, Xa_tau, tau)   # (system, ens)
        xbar_a_end = jnp.mean(Xa_end, axis=1)
        ya_end = jax.vmap(lambda i: Hs[i] @ xbar_a_end)(
                jnp.arange(Hs.shape[0])).reshape(-1).astype(dtype)
        # End-of-window analysis obs-space perturbations (obs x ens): drives
        # ``obs_space_spread_analysis_end`` (the spread of the ensemble handed
        # to the next cycle -- NOT ``obs_space_spread_background`` at tau).
        Ya_end = jax.vmap(lambda i: Hs[i] @ Xa_end)(
                jnp.arange(Hs.shape[0])).reshape(-1, self.ensemble_dim)
        ens_obs_end = (Ya_end - jnp.mean(Ya_end, axis=1)[:, None]).astype(dtype)
        end_active = active & (jnp.repeat(owi, Hs.shape[1]) == end_idx)

        # tau-restricted O-F/O-A comparison pair (obs at the analysis time
        # only): the like-for-like fit diagnostic where O-A <= O-F holds.  The
        # universal o_minus_f_rms/o_minus_a_rms above cover ALL obs (O-F at the
        # appropriate time) for method comparison; this pair is ONLY for the
        # direct O-F vs O-A comparison at tau.
        tau_active = active & (jnp.repeat(owi, Hs.shape[1]) == tau)
        cmp = dac_utils._obs_space_metrics(
                Y, yf_bar, ya_bar, tau_active, sigma2_diag,
                dtype=dtype)

        out = dac_utils._obs_space_metrics(
                Y, yf_bar, ya_bar, active, sigma2_diag,
                ens_obs=ens_obs,
                return_per_obs=(self._metrics_mode == "debug"), dtype=dtype,
                Hxa_end_mean=ya_end, end_active_mask=end_active,
                ens_obs_end=ens_obs_end)
        out["o_minus_f_rms_at_tau"] = cmp["o_minus_f_rms"]
        out["o_minus_a_rms_at_tau"] = cmp["o_minus_a_rms"]
        return out

    def _xa_end_ensemble(self, cur_state, Xa_tau, tau):
        """Full analysis ENSEMBLE propagated to the window end (next-cycle IC).

        ``Xa_tau`` is the analysis ensemble ``(system, ens)`` at the analysis
        time ``tau``.  Re-forecasts every member from ``tau`` to the window end
        and returns the end-of-window analysis ensemble ``(system, ens)`` --
        the ensemble that seeds the next cycle; when ``tau`` is already the last
        window index the analysis IS the end ensemble (no roll).
        """
        xdims = cur_state['x'].dims
        if tau == self.steps_per_window - 1:
            return Xa_tau
        Xa_o = Xa_tau.T if xdims[0] == 'ensemble' else Xa_tau
        ana_tau_ds = cur_state.assign(x=(xdims, Xa_o))
        end_ds, _ = self._step_forecast(
                ana_tau_ds, n_steps=self.steps_per_window - tau)
        Xa_end = end_ds.to_stacked_array('system', ['ensemble']).data.T
        return jnp.asarray(Xa_end)                          # (system, ens)

    def _hxa_end_mean_ensemble(self, cur_state, Xa_tau, tau):
        """Ensemble analysis MEAN propagated to the window end (next-cycle IC).

        Thin wrapper over :meth:`_xa_end_ensemble` returning the ensemble-mean
        state ``(system,)`` at the window end.
        """
        return jnp.mean(self._xa_end_ensemble(cur_state, Xa_tau, tau), axis=1)

    def _update_ns_diagnostics(self, ns_resid) -> None:
        """Store the Newton-Schulz convergence residual + warn if too large.

        ``ns_resid`` is ``||Z A Z - I||_F / sqrt(K)`` (finite on the NS path,
        ``NaN`` on the eigh paths).  Follows :meth:`LETKF._update_diagnostics`:
        only stores a CONCRETE value (silently skips under a JAX trace, e.g.
        inside ``jax.lax.scan``, so no tracer leaks onto ``self`` and no warning
        fires per-trace); callers wanting the diagnostic evaluate the analysis
        eagerly.  When concrete and above ``self.ns_resid_warn``, emits a
        warning suggesting a larger ``ns_iters``.
        """
        if self.eigh_impl != "newton_schulz":
            return
        try:
            resid = float(ns_resid)
        except (jax.errors.TracerArrayConversionError,
                jax.errors.ConcretizationTypeError):
            return
        self.ns_resid = resid
        if np.isfinite(resid) and resid > self.ns_resid_warn:
            warnings.warn(
                f"Newton-Schulz SPD solve residual {resid:.2e} exceeds "
                f"ns_resid_warn={self.ns_resid_warn:.2e}; the ETKF transform "
                f"may be under-converged -- increase ns_iters (currently "
                f"{self.ns_iters}).",
                stacklevel=2)

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

        ana_ds = Xb_ds.assign(x=(['ensemble', 'i'], Xa.T))
        if not self._return_metrics:
            return ana_ds
        dtype = Xa.dtype
        y = jnp.asarray(obs_values, dtype).reshape(-1)
        H = jnp.asarray(H, dtype)
        xb_mean = jnp.mean(Xb, axis=1)
        xa_mean = jnp.mean(Xa, axis=1)
        Hxb_mean = H @ xb_mean
        Hxa_mean = H @ xa_mean
        ens_obs = H @ (Xb - xb_mean[:, None])            # (obs_dim, ens)
        active = (obs_time_mask.reshape(-1).astype(bool)
                  & obs_loc_mask.reshape(-1).astype(bool))
        sigma2_diag = jnp.diag(jnp.asarray(R, dtype))
        metrics = dac_utils._obs_space_metrics(
                y, Hxb_mean, Hxa_mean, active, sigma2_diag, ens_obs=ens_obs,
                return_per_obs=(self._metrics_mode == "debug"), dtype=dtype)
        return ana_ds, metrics
