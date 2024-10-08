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
import xarray as xr
import xarray_jax as xj

from dabench import dacycler
import dabench.dacycler._utils as dac_utils


class Var4D(dacycler.DACycler):
    """Class for building 4D DA Cycler

    Attributes:
        system_dim (int): System dimension.
        delta_t (float): The timestep of the model (assumed uniform)
        model_obj (dabench.Model): Forecast model object.
        in_4d (bool): True for 4D data assimilation techniques (e.g. 4DVar).
            Always True for Var4D.
        ensemble (bool): True for ensemble-based data assimilation techniques
            (ETKF). Always False for Var4D.
        B (ndarray): Initial / static background error covariance. Shape:
            (system_dim, system_dim). If not provided, will be calculated
            automatically.
        R (ndarray): Observation error covariance matrix. Shape
            (obs_dim, obs_dim). If not provided, will be calculated
            automatically.
        H (ndarray): Observation operator with shape: (obs_dim, system_dim).
            If not provided will be calculated automatically.
        h (function): Optional observation operator as function. More flexible
            (allows for more complex observation operator). Default is None.
        solver (str): Name of solver to use. Default is 'bicgstab'.
        n_outer_loops (int): Number of times to run through outer loop over
            4DVar. Increasing this may result in higher accuracy but slower
            performance. Default is 1.
        steps_per_window (int): Number of timesteps per analysis window.
            If None (default), will calculate automatically based on delta_t
            and .cycle() analysis_window length.
        obs_window_indices (list): Timestep indices where observations fall
            within each analysis window. For example, if analysis window is
            0 - 0.05 with delta_t = 0.01 and observations fall at 0, 0.01,
            0.02, 0.03, 0.04, and 0.05, obs_window_indices =
            [0, 1, 2, 3, 4, 5]. If None (default), will calculate
            automatically.
    """

    def __init__(self,
                 system_dim=None,
                 delta_t=None,
                 model_obj=None,
                 B=None,
                 R=None,
                 H=None,
                 h=None,
                 solver='bicgstab',
                 n_outer_loops=1,
                 steps_per_window=1,
                 obs_window_indices=None,
                 analysis_time_in_window=0,
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
                         B=B, R=R, H=H, h=h,
                         analysis_time_in_window=analysis_time_in_window)

    def _calc_default_H(self, obs_loc_indices):
        Hs = jnp.zeros((obs_loc_indices.shape[0], obs_loc_indices.shape[1],
                        self.system_dim),
                       dtype=int)
        for i in range(Hs.shape[0]):
            Hs = Hs.at[i, jnp.arange(Hs.shape[1]), obs_loc_indices
                       ].set(1)
        return Hs

    def _calc_default_R(self, obs_values, obs_error_sd):
        return jnp.identity(obs_values[0].shape[0])*(obs_error_sd**2)

    def _calc_J_term(self, H, M, Rinv, y, x):
        # The Jb Term (A)
        HM = H @ M
        MtHtRinv = HM.T @ Rinv

        # The Jo Term (b)
        D = (y - (H @ x))
        return MtHtRinv @ HM,  MtHtRinv @ D[:, None]

    @partial(jax.jit, static_argnums=[0, 1])
    def _innerloop_4d(self, system_dim, x, xb0, obs_vals, Hs, B, Rinv, M,
                      obs_window_indices, obs_time_mask):
        """4DVar innerloop"""
        x0_last = x.isel(time=0)
        x = x.to_stacked_array('system',['time'])

        # Set up Variables
        SumMtHtRinvHM = jnp.zeros_like(B)             # A input
        SumMtHtRinvD = jnp.zeros((system_dim, 1))     # b input

        # Loop over observations
        for i, j in enumerate(obs_window_indices):
            Jb, Jo = jax.lax.cond(
                    obs_time_mask.at[i].get(mode='fill', fill_value=0),
                    lambda: self._calc_J_term(Hs.at[i].get(mode='clip'), M.data[j],
                                              Rinv, obs_vals[i], x.data[j]),
                    lambda: (jnp.zeros_like(SumMtHtRinvHM),
                             jnp.zeros_like(SumMtHtRinvD))
                    )
            SumMtHtRinvHM += Jb
            SumMtHtRinvD += Jo
        # Compute initial departure
        db0 = (xb0 - x0_last).to_stacked_array('system',[]).data

        # Solve Ax=b for the initial perturbation
        dx0 = self._solve(db0, SumMtHtRinvHM, SumMtHtRinvD, B)

        # New x0 guess is the last guess plus the analyzed delta
        x0_new = x0_last + dx0.ravel()

        return x0_new

    def _make_outerloop_4d(self, xb0,  Hs, B, Rinv,
                           obs_values, obs_window_indices, obs_time_mask,
                           n_steps):

        def _outerloop_4d(x0, _):
            # Get TLM and current forecast trajectory
            # Based on current best guess for x0
            x0 = x0.to_xarray()
            x, M = self.model_obj.compute_tlm(
                n_steps=n_steps,
                state_vec=x0
            )

            # 4D-Var inner loop
            x0_new = self._innerloop_4d(self.system_dim,
                                    x, xb0, obs_values,
                                    Hs, B, Rinv, M,
                                    obs_window_indices, 
                                    obs_time_mask)

            return xj.from_xarray(x0_new.assign_coords(x0.coords)), x0

        return _outerloop_4d

    @partial(jax.jit, static_argnums=0)
    def _solve(self, db0, SumMtHtRinvHM, SumMtHtRinvD, B):
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

    def _cycle_obsop(self, xb0, obs_values, obs_loc_indices,
                     obs_time_mask, obs_loc_mask,
                     H=None, h=None, R=None, B=None, obs_window_indices=None):
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
        x0_new = deepcopy(xb0)

        outerloop_4d_func = self._make_outerloop_4d(
                xb0,  Hs, B, Rinv, obs_values, obs_window_indices,
                obs_time_mask, self.steps_per_window)

        x0_new, all_x0s = jax.lax.scan(outerloop_4d_func, init=xj.from_xarray(x0_new),
                xs=None, length=self.n_outer_loops)

        return x0_new.to_xarray()
