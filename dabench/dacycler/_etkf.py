"""Class for Ensemble Transform Kalman Filter (ETKF) DA Class"""

import warnings

import numpy as np
import jax
import jax.numpy as jnp
import xarray as xr
from dabench import _xarray_jax as xj
from typing import Callable

from dabench import dacycler
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
            ``eigh_impl="newton_schulz"``).  Default 20 -- at the fp64 round-off
            floor for a well-conditioned (kappa<1e3) 64x64 SPD transform.
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
                 ns_iters: int = 20,
                 ns_resid_warn: float = 1e-6
                 ):

        self.ensemble_dim = ensemble_dim
        self.multiplicative_inflation = multiplicative_inflation
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
        Z = jax.random.normal(key, (ensemble_dim, ensemble_dim))
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

        return Xb_ds.assign(x=(['ensemble','i'], Xa.T))
