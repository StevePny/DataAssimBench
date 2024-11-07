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
import lineax as lx

from dabench import dacycler, vector
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

    def _calc_default_B(self):
        return jnp.identity(self.system_dim)

    def _make_outerloop_4d(self, xb0,  Hs, B, Rinv,
                           obs_values, obs_window_indices, obs_time_mask,
                           n_steps):

        def _outerloop_4d(x0, _):
            # Get TLM and current forecast trajectory
            # Based on current best guess for x0
            M, x = self.model_obj.compute_tlm(
                n_steps=n_steps,
                state_vec=vector.StateVector(values=x0,
                                             store_as_jax=True)
            )

            # 4D-Var inner loop
            x0 = self._innerloop_4d(self.system_dim,
                                    x, xb0, obs_values,
                                    Hs, B, Rinv, M,
                                    obs_window_indices, 
                                    obs_time_mask)

            return x0, x0

        return _outerloop_4d

    def _cycle_obsop(self, xb0, obs_values, obs_loc_indices, obs_error_sd,
                     obs_window_indices, obs_time_mask, obs_loc_mask,
                     H=None, h=None, R=None, B=None,
                     n_steps=1):
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
                R = self._calc_default_R(obs_values, obs_error_sd)
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
        x0 = deepcopy(xb0)

        outerloop_4d_func = self._make_outerloop_4d(
                xb0,  Hs, B, Rinv, obs_values, obs_window_indices,
                obs_time_mask, n_steps)

        x0, all_x0s = jax.lax.scan(outerloop_4d_func, init=x0,
                xs=None, length=self.n_outer_loops)

        # forecast
        x = self.step_forecast(
            n_steps=n_steps,
            x0=vector.StateVector(values=x0, store_as_jax=True)
        ).values

        return x

    def step_cycle(self, x0, yo, obs_time_mask, obs_loc_mask,
                   obs_window_indices, H=None, h=None, R=None, B=None,
                   n_steps=1):
        """Perform one step of DA Cycle"""
        if H is not None or h is None:
            return self._cycle_obsop(
                    x0.values, yo.values, yo.location_indices, yo.error_sd,
                    obs_loc_mask=obs_loc_mask, obs_time_mask=obs_time_mask,
                    obs_window_indices=obs_window_indices,
                    H=H, R=R, B=B, 
                    n_steps=n_steps)
        else:
            return self._cycle_obsop(
                    x0.values, yo.values, yo.location_indices, yo.error_sd, h=h,
                    R=R, B=B, obs_window_indices=obs_window_indices,
                    n_steps=n_steps)

    def step_forecast(self, x0, n_steps=1):
        """Perform forecast using model object"""
        if 'n_steps' in inspect.getfullargspec(self.model_obj.forecast).args:
            return self.model_obj.forecast(x0, n_steps=n_steps)
        else:
            if n_steps == 1:
                return self.model_obj.forecast(x0)
            else:
                out = [x0]
                xi = x0
                for s in range(n_steps):
                    xi = self.model.forecast(xi)
                    out.append(xi)
                return vector.StateVector(jnp.vstack(xi), store_as_jax=True)


    def _calc_Jo_term(self, H, M, Rinv, y, x):
        # The Jo Term (b)
        D = (y - (H @ x))
        return M.T @ (H.T @ (Rinv @ D[:, None]))

    def _calc_J_term(self, H, M, Rinv, y, x):
        # The Jb Term (A)
        HM = H @ M
        MtHtRinv = HM.T @ Rinv

        # The Jo Term (b)
        D = (y - (H.matrix @ x))
        return MtHtRinv @ HM,  (M.matrix.T @ (H.matrix.T @(Rinv.matrix @ D[:, None])))


    @partial(jax.jit, static_argnums=[0, 1])
    def _innerloop_4d(self, system_dim, x, xb0, obs_vals, Hs, B, Rinv, M,
                      obs_window_indices, obs_time_mask):
        """4DVar innerloop"""
        x0_last = x[0]

        # Set up Variables
        SumMtHtRinvHM = lx.MatrixLinearOperator(jnp.zeros_like(B))             # A input
        SumMtHtRinvD = jnp.zeros((system_dim, 1))     # b input

        # Compute initial departure
        db0 = xb0 - x0_last

        # Loop over observations
        for i, j in enumerate(obs_window_indices):
            valid_obs = obs_time_mask.at[i].get(mode='fill', fill_value=0)
            Jb, Jo = self._calc_J_term(
                        lx.MatrixLinearOperator(Hs.at[i].get(mode='clip')),
                        lx.MatrixLinearOperator(M[j]),
                        lx.MatrixLinearOperator(Rinv), obs_vals[i], x[j]
                        )
            # Jb, Jo = jax.lax.cond(
            #         obs_time_mask.at[i].get(mode='fill', fill_value=0),
            #         lambda: self._calc_J_term(
            #             lx.MatrixLinearOperator(Hs.at[i].get(mode='clip')),
            #             lx.MatrixLinearOperator(M[j]),
            #             lx.MatrixLinearOperator(Rinv), obs_vals[i], x[j]),
            #         lambda: (lx.MatrixLinearOperator(jnp.zeros_like(SumMtHtRinvHM)),
            #                  jnp.zeros_like(SumMtHtRinvD))
            #         )
            SumMtHtRinvHM = SumMtHtRinvHM + valid_obs * Jb
            SumMtHtRinvD = SumMtHtRinvD + valid_obs * Jo

        # Solve Ax=b for the initial perturbation
        dx0 = self._solve_lx(db0, SumMtHtRinvHM, SumMtHtRinvD, lx.MatrixLinearOperator(B))
        # dx0 = self._solve_linop(db0, Hs, Rinv, M, SumMtHtRinvD, B,
        #                         obs_window_indices, obs_time_mask)

        # New x0 guess is the last guess plus the analyzed delta
        x0_new = x0_last + dx0.ravel()

        return x0_new

    @partial(jax.jit, static_argnums=0)
    def _solve_linop(self, db0, Hs, Rinv, M, SumMtHtRinvD, B,
                     obs_window_indices, obs_time_mask):
        """Solve the 4D-Var linear optimization

        Notes:
            Solves Ax=b for x when:
            A = B^{-1} + SumMtHtRinvHM
            b = SumMtHtRinvD + db0[:,None]
        """

        # Initialize b array
        system_dim = B.shape[1]

        def MtHtRinvHM(i, j, x):
            H_temp = Hs.at[i].get(mode='clip')
            M_temp = M[j]
            return M_temp.T @ (H_temp.T @ (Rinv @ (H_temp @ (M_temp @ x))))

        def A_func(x):
            intermediate_x = jnp.zeros_like(x)
            for i, j in enumerate(obs_window_indices):
                Jb = jax.lax.cond(
                        obs_time_mask.at[i].get(mode='fill', fill_value=0),
                        lambda: MtHtRinvHM(i, j, x),
                        lambda: jnp.zeros_like(intermediate_x)
                )
                intermediate_x += Jb
            return x + B @ intermediate_x


        # Solve 4D-Var cost function
        if self.solver == 'bicgstab':
            # Compute A,b inputs to linear minimizer
            b1 = B @ SumMtHtRinvD + db0[:, None]

            dx0, _ = bicgstab(A_func, b1, x0=jnp.zeros((system_dim, 1)), tol=1e-05)

        else:
            raise ValueError("Solver not recognized. Options: 'bicgstab'")

        return dx0

    @partial(jax.jit, static_argnums=0)
    def _solve_lx(self, db0, SumMtHtRinvHM, SumMtHtRinvD, B):
        """Solve the 4D-Var linear optimization

        Notes:
            Solves Ax=b for x when:
            A = B^{-1} + SumMtHtRinvHM
            b = SumMtHtRinvD + db0[:,None]
        """

        # Set identity matrix
        I_mat = lx.MatrixLinearOperator(jnp.identity(B.in_size()))

        # Solve 4D-Var cost function
        if self.solver == 'bicgstab':
            # Compute A,b inputs to linear minimizer
            b1 = B.matrix @ SumMtHtRinvD + db0[:, None]

            A = I_mat + B @ SumMtHtRinvHM
            solver = lx.BiCGStab(rtol=1e-5, atol=1e-5)

            dx0 = lx.linear_solve(A, b1[:,0], solver)

        else:
            raise ValueError("Solver not recognized. Options: 'bicgstab'")

        return dx0.value

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

    def _cycle_and_forecast(self, cur_state_vals_time_tuple, filtered_idx):
        cur_state_vals, cur_time = cur_state_vals_time_tuple
        obs_error_sd = self._obs_error_sd

        # Calculate obs_time_mask and restore filtered_idx to original values
        obs_time_mask = filtered_idx > 0
        filtered_idx = filtered_idx - 1

        cur_obs_vals = jnp.array(self._obs_vector.values).at[filtered_idx].get()
        cur_obs_loc_indices = jnp.array(self._obs_vector.location_indices).at[filtered_idx].get()
        cur_obs_times = jnp.array(self._obs_vector.times).at[filtered_idx].get()
        cur_obs_loc_mask = jnp.array(self._obs_loc_masks).at[filtered_idx].get().astype(bool)

        # Calculate obs window indices: closest model timesteps that match obs
        obs_window_indices = jax.lax.cond(
            self.obs_window_indices is None,
            lambda: jnp.array([
                jnp.argmin(
                    jnp.abs(obs_time - (cur_time + self._model_timesteps))
                    ) for obs_time in cur_obs_times
            ]),
            lambda: jnp.array(self.obs_window_indices)
            )

        analysis = self.step_cycle(
                vector.StateVector(values=cur_state_vals, store_as_jax=True),
                vector.ObsVector(values=cur_obs_vals,
                                 location_indices=cur_obs_loc_indices,
                                 error_sd=obs_error_sd,
                                 store_as_jax=True),
                obs_time_mask=obs_time_mask,
                obs_loc_mask=cur_obs_loc_mask,
                n_steps=self.steps_per_window,
                obs_window_indices=obs_window_indices)
        new_time = cur_time + self.analysis_window

        return (analysis[-1], new_time), analysis[:-1]

    def cycle(self,
              input_state,
              start_time,
              obs_vector,
              obs_error_sd,
              n_cycles,
              analysis_window,
              analysis_time_in_window=0,
              return_forecast=False):
        """Perform DA cycle repeatedly, including analysis and forecast

        Args:
            input_state (vector.StateVector): Input state.
            start_time (float or datetime-like): Starting time.
            obs_vector (vector.ObsVector): Observations vector.
            obs_error_sd (float): Standard deviation of observation error.
                Typically not known, so provide a best-guess.
            n_cycles (int): Number of analysis cycles to run, each of length
                analysis_window.
            analysis_window (float): Length of time window from which to gather
                observations for each DA Cycle, in model time units.
            analysis_time_in_window (float): At what time within analysis_window
                to perform analysis. For example, 0.0 is the start of the
                window. Default is 0, the start of the window.
            return_forecast (bool): If True, returns forecast at each model
                timestep. If False, returns only analyses, one per analysis
                cycle. Default is False.

        Returns:
            vector.StateVector of analyses and times.
        """
        if (not obs_vector.stationary_observers and
                (self.H is not None or self.h is not None)):
            warnings.warn(
                "Provided obs vector has nonstationary observers. When"
                " providing a custom obs operator (H/h), the Var4DBackprop"
                "DA cycler may not function properly. If you encounter "
                "errors, try again with an observer where"
                "stationary_observers=True or without specifying H or h (a "
                "default H matrix will be used to map observations to system "
                "space)."
            )
        self.analysis_window = analysis_window

        # If don't specify analysis_time_in_window, is assumed to be middle
        if analysis_time_in_window is None:
            analysis_time_in_window = self.analysis_window/2

        # Time offset from middle of time window, for gathering observations
        _time_offset = (analysis_window/2) - analysis_time_in_window

        # Set up for jax.lax.scan, which is very fast
        all_times = dac_utils._get_all_times(start_time, analysis_window,
                                             n_cycles)

        if self.steps_per_window is None:
            self.steps_per_window = round(analysis_window/self.delta_t) + 1
        self._model_timesteps = jnp.arange(self.steps_per_window)*self.delta_t

        # Get the obs vectors for each analysis window
        all_filtered_idx = dac_utils._get_obs_indices(
            obs_times=obs_vector.times,
            analysis_times=all_times+_time_offset,
            start_inclusive=True,
            end_inclusive=True,
            analysis_window=analysis_window
        )

        all_filtered_padded = dac_utils._pad_time_indices(all_filtered_idx)

        self._obs_vector = obs_vector
        self._obs_error_sd = obs_error_sd

        # Padding observations
        if obs_vector.stationary_observers:
            self._obs_loc_masks = jnp.ones(obs_vector.values.shape, dtype=bool)
        else:
            obs_vals, obs_locs, obs_loc_masks = dac_utils._pad_obs_locs(
                    obs_vector)
            self._obs_vector.values = obs_vals
            self._obs_vector.location_indices = obs_locs
            self._obs_loc_masks = jnp.array(obs_loc_masks)

        cur_state, all_values = jax.lax.scan(
                self._cycle_and_forecast,
                init=(input_state.values, start_time),
                xs=all_filtered_padded)

        if return_forecast:
            all_times_forecast = jnp.arange(
                0,
                n_cycles*analysis_window,
                self.delta_t
                ) + start_time
            return vector.StateVector(values=jnp.concatenate(all_values),
                                      times=all_times_forecast)
        else:
            return vector.StateVector(values=jnp.vstack([
                forecast[0] for forecast in all_values]
                ),
                                      times=all_times)
