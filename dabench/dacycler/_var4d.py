"""Class for Var 4D Data Assimilation Cycler object"""

import inspect
import warnings

import numpy as np
import jax.numpy as jnp
import jax.scipy as jscipy
from jax import grad
import jax
from jax.scipy.sparse.linalg import bicgstab
from copy import deepcopy
from functools import partial
from typing import Callable
import xarray as xr
import xarray_jax as xj

from dabench import dacycler
from dabench.model import Model
import dabench.dacycler._utils as dac_utils


# For typing
ArrayLike = np.ndarray | jax.Array
XarrayDatasetLike = xr.Dataset | xj.XjDataset

class Var4D(dacycler.DACycler):
    """Class for building 4D DA Cycler

    Attributes:
        system_dim: System dimension.
        delta_t: The timestep of the model (assumed uniform)
        model_obj: Forecast model object.
        in_4d: True for 4D data assimilation techniques (e.g. 4DVar).
            Always True for Var4D.
        ensemble: True for ensemble-based data assimilation techniques
            (ETKF). Always False for Var4D.
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
        solver: Name of solver to use. Default is 'bicgstab'.
        n_outer_loops: Number of times to run through outer loop over
            4DVar. Increasing this may result in higher accuracy but slower
            performance. Default is 1.
        steps_per_window: Number of timesteps per analysis window.
            If None (default), will calculate automatically based on delta_t
            and .cycle() analysis_window length.
        obs_window_indices: Timestep indices where observations fall
            within each analysis window. For example, if analysis window is
            0 - 0.05 with delta_t = 0.01 and observations fall at 0, 0.01,
            0.02, 0.03, 0.04, and 0.05, obs_window_indices =
            [0, 1, 2, 3, 4, 5]. If None (default), will calculate
            automatically.
    """

    def __init__(self,
                 system_dim: int,
                 delta_t: float,
                 model_obj: Model,
                 B: ArrayLike | None = None,
                 R: ArrayLike | None = None,
                 H: ArrayLike | None = None,
                 h: Callable | None = None,
                 solver: str = 'bicgstab',
                 n_outer_loops: int = 1,
                 steps_per_window: int = 1,
                 obs_window_indices: ArrayLike | None = None,
                 **kwargs
                 ):

        self.steps_per_window = steps_per_window
        self.obs_window_indices = obs_window_indices
        self.n_outer_loops = n_outer_loops
        self.solver = solver

        # Var4D requires H to be a JAX array
        if H is not None:
            H = jnp.array(H)

        super().__init__(system_dim=system_dim,
                         delta_t=delta_t,
                         model_obj=model_obj,
                         in_4d=True,
                         ensemble=False,
                         B=B, R=R, H=H, h=h)

    def _calc_default_H(self,
                        obs_loc_indices: ArrayLike
                        ) -> jax.Array:
        Hs = jnp.zeros((obs_loc_indices.shape[0], obs_loc_indices.shape[1],
                        self.system_dim),
                       dtype=int)
        for i in range(Hs.shape[0]):
            Hs = Hs.at[i, jnp.arange(Hs.shape[1]), obs_loc_indices
                       ].set(1)
        return Hs

    def _calc_default_R(self,
                        obs_values: ArrayLike,
                        obs_error_sd: float
                        ) -> jax.Array:
        return jnp.identity(obs_values[0].shape[0])*(obs_error_sd**2)

    def _calc_J_term(self,
                     H: ArrayLike,
                     M: ArrayLike,
                     Rinv: ArrayLike,
                     y: ArrayLike,
                     x: ArrayLike
                     ) -> jax.Array:
        # The Jb Term (A)
        HM = H @ M
        MtHtRinv = HM.T @ Rinv

        # The Jo Term (b)
        D = (y - (H @ x))
        return MtHtRinv @ HM,  MtHtRinv @ D[:, None]

    @partial(jax.jit, static_argnums=[0, 1])
    def _innerloop_4d(self,
                      system_dim: int,
                      Xb_ds: XarrayDatasetLike,
                      xb0_ds: XarrayDatasetLike,
                      obs_vals: ArrayLike,
                      Hs: ArrayLike,
                      B: ArrayLike,
                      Rinv: ArrayLike,
                      M: ArrayLike,
                      obs_window_indices: ArrayLike | list,
                      obs_time_mask: ArrayLike
                      ) -> XarrayDatasetLike:
        """4DVar innerloop"""
        x0_prev_ds = Xb_ds.isel(time=0)
        Xb_ar = Xb_ds.to_stacked_array('system',['time'])

        # Set up Variables
        SumMtHtRinvHM = jnp.zeros_like(B)             # A input
        SumMtHtRinvD = jnp.zeros((system_dim, 1))     # b input

        # Loop over observations
        for i, j in enumerate(obs_window_indices):
            Jb, Jo = jax.lax.cond(
                    obs_time_mask.at[i].get(mode='fill', fill_value=0),
                    lambda: self._calc_J_term(
                        Hs.at[i].get(mode='clip'),
                        M.data[j],
                        Rinv, obs_vals[i], Xb_ar.data[j]),
                    lambda: (jnp.zeros_like(SumMtHtRinvHM),
                             jnp.zeros_like(SumMtHtRinvD))
                    )
            SumMtHtRinvHM += Jb
            SumMtHtRinvD += Jo
        # Compute initial departure
        db0 = (xb0_ds - x0_prev_ds).to_stacked_array('system',[]).data

        # Solve Ax=b for the initial perturbation
        dx0 = self._solve(db0, SumMtHtRinvHM, SumMtHtRinvD, B)

        # New x0 guess is the last guess plus the analyzed delta
        x0_new_ds = x0_prev_ds + dx0.ravel()

        return x0_new_ds

    def _make_outerloop_4d(self,
                           xb0_ds: XarrayDatasetLike,
                           Hs: ArrayLike,
                           B: ArrayLike,
                           Rinv: ArrayLike,
                           obs_values: ArrayLike,
                           obs_window_indices: ArrayLike | list,
                           obs_time_mask: ArrayLike,
                           n_steps: int
                           ) -> Callable:

        def _outerloop_4d(x0_ds: XarrayDatasetLike,
                          _: None
                          ) -> tuple[XarrayDatasetLike, XarrayDatasetLike]:
            # Get TLM and current forecast trajectory
            # Based on current best guess for x0
            x0_ds = x0_ds.to_xarray()
            xb_ds, M = self.model_obj.compute_tlm(
                n_steps=n_steps,
                state_vec=x0_ds
            )

            # 4D-Var inner loop
            x0_new_ds = self._innerloop_4d(
                self.system_dim, xb_ds, xb0_ds, obs_values,
                Hs, B, Rinv, M, obs_window_indices, obs_time_mask
                )

            return xj.from_xarray(x0_new_ds.assign_coords(x0_ds.coords)), x0_ds

        return _outerloop_4d

    @partial(jax.jit, static_argnums=0)
    def _solve(self,
               db0: ArrayLike,
               SumMtHtRinvHM: ArrayLike,
               SumMtHtRinvD: ArrayLike,
               B: ArrayLike
               ) -> jax.Array:
        """Solve the 4D-Var linear optimization

        Notes:
            Solves Ax=b for x when:
            A = B^{-1} + SumMtHtRinvHM
            b = SumMtHtRinvD + db0[:,None]
        """

        # Initialize b array
        system_dim = B.shape[1]

        # Set identity matrix
        I_mat = jnp.identity(B.shape[0])

        # Solve 4D-Var cost function
        if self.solver == 'bicgstab':
            # Compute A,b inputs to linear minimizer
            b1 = B @ SumMtHtRinvD + db0[:, None]

            A = I_mat + B @ SumMtHtRinvHM

            dx0, _ = bicgstab(A, b1, x0=jnp.zeros((system_dim, 1)), tol=1e-05)

        else:
            raise ValueError("Solver not recognized. Options: 'bicgstab'")

        return dx0

    def _cycle_obsop(self,
                     x0_ds: XarrayDatasetLike,
                     obs_values: ArrayLike,
                     obs_loc_indices: ArrayLike,
                     obs_time_mask: ArrayLike,
                     obs_loc_mask: ArrayLike,
                     H: ArrayLike | None = None,
                     h: Callable | None = None,
                     R: ArrayLike | None = None,
                     B: ArrayLike | None = None,
                     obs_window_indices = ArrayLike | list | None
                     ) -> XarrayDatasetLike:
        if H is None and h is None:
            if self.H is None:
                if self.h is None:
                    H = self._calc_default_H(obs_loc_indices)
                    # Apply obs loc mask
                    # NOTE: nonstationary observer case runs MUCH slower. Not sure why
                    # Ideally, this conditional would not be necessary, but this is a
                    # workaround to prevent slowing down stationary observer case.
                    Hs = jax.lax.cond(
                        self._obs_vector.stationary_observers,
                        lambda: H,
                        lambda: (obs_loc_mask[:, :, jnp.newaxis] * H))
                else:
                    h = self.h
            else:
                # Assumes self.H is for a single timestep
                H = self.H[jnp.newaxis]
                Hs = jax.lax.cond(
                    self._obs_vector.stationary_observers,
                    lambda: jnp.repeat(H, obs_values.shape[0], axis=0),
                    lambda: (obs_loc_mask[:, :, jnp.newaxis] * H))

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

        # Static Variables
        Rinv = jscipy.linalg.inv(R)

        # Best guess for x0 starts as background
        x0_new_ds = deepcopy(x0_ds)

        outerloop_4d_func = self._make_outerloop_4d(
                x0_ds, Hs, B, Rinv, obs_values, obs_window_indices,
                obs_time_mask, self.steps_per_window)

        x0_new_ds, all_x0s = jax.lax.scan(outerloop_4d_func, init=xj.from_xarray(x0_new_ds),
                xs=None, length=self.n_outer_loops)

        return x0_new_ds.to_xarray()
