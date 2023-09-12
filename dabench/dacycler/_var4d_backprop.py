"""Class for 4D Backpropagation  Data Assimilation Cycler object"""

import inspect

import numpy as np
import jax.numpy as jnp
import jax.scipy as jscipy
from jax import value_and_grad, grad
import jax

from dabench import dacycler, vector


class Var4DBackprop(dacycler.DACycler):
    """Class for building Backpropagation 4D DA Cycler

    Attributes:
    """

    def __init__(self,
                 system_dim=None,
                 delta_t=None,
                 model_obj=None,
                 B=None,
                 R=None,
                 h=None,
                 H=None,
                 learning_rate=1e-5,
                 num_epochs=20,
                 steps_per_window=1,
                 obs_window_indices=[0],
                 **kwargs
                 ):

        self.h = h
        self.H = H
        self.R = R
        self.B = B
        self.num_epochs = num_epochs
        self.learning_rate = learning_rate
        self.steps_per_window = steps_per_window
        self.obs_window_indices = obs_window_indices

        super().__init__(system_dim=system_dim,
                         delta_t=delta_t,
                         model_obj=model_obj,
                         in_4d=True,
                         ensemble=False)

    def _calc_default_H(self, obs_values, obs_loc_indices):
        H = jnp.zeros((obs_values.flatten().shape[0], self.system_dim))
        H = H.at[jnp.arange(H.shape[0]), obs_loc_indices.flatten()
                 ].set(1)
        return H

    def _calc_default_R(self, obs_values, obs_error_sd):
        return jnp.identity(obs_values.flatten().shape[0])*(obs_error_sd**2)

    def _calc_default_B(self):
        return jnp.identity(self.system_dim)

    def _make_loss(self, obs_vals, H, B, R, time_sel_matrix, rb0, n_steps):
        """Define loss function based on 4dvar cost"""
        Rinv = jscipy.linalg.inv(R)
        Binv = jscipy.linalg.inv(B)

        def loss_4dvarcost(r0):
            # Make prediction based on current r
            pred_r = self.step_forecast(
                    vector.StateVector(values=r0, store_as_jax=True),
                    n_steps=n_steps).values

            # Apply observation operator to map to obs spcae
            pred_obs = time_sel_matrix @ pred_r @ H

            # Calculate observation term of J_0
            resid = pred_obs.ravel() - obs_vals.ravel()
            obs_term = 0.5*np.sum(resid.T @ Rinv @ resid)

            # Calculate initial departure term of J_0 based on original x0
            db0 = pred_r[0].ravel() - rb0.ravel()
            initial_term = 0.5*(db0.T @ Binv @ db0)

            # Cost is the sum of the two terms
            return initial_term + obs_term

        return loss_4dvarcost

    def _calc_time_sel_matrix(self, obs_steps_inds, n_pred_steps):
        time_sel_matrix = jnp.zeros((len(obs_steps_inds), n_pred_steps))
        time_sel_matrix = time_sel_matrix.at[
                jnp.arange(time_sel_matrix.shape[0]), obs_steps_inds].set(1)
        return time_sel_matrix

    def _make_backprop_epoch(self, loss_func):

        def _backprop_epoch(i, r0):
            dr0 = grad(loss_func, argnums=0)(r0)
            r0 -= self.learning_rate*dr0
            return r0

        return _backprop_epoch

    def _cycle_obsop(self, xb, obs_values, obs_loc_indices, obs_error_sd,
                     H=None, h=None, R=None, B=None, time_sel_matrix=None, n_steps=1):
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
                R = self._calc_default_R(obs_values, obs_error_sd)
            else:
                R = self.R
        if B is None:
            if self.B is None:
                B = self._calc_default_B()
            else:
                B = self.B

        r0 = xb
        loss_func = self._make_loss(obs_values, H, B, R, time_sel_matrix,
                                    rb0=r0, n_steps=n_steps)
        backprop_epoch_func = self._make_backprop_epoch(loss_func)
        r0 = jax.lax.fori_loop(0, self.num_epochs, backprop_epoch_func, r0)

        ra = self.step_forecast(
                vector.StateVector(values=r0, store_as_jax=True),
                n_steps=n_steps)

        return ra, None

    def step_cycle(self, xb, yo, H=None, h=None, R=None, B=None, n_steps=1,
                   obs_window_indices=[0]):
        """Perform one step of DA Cycle

        Args:
            xb:
            yo:
            H


        Returns:
            vector.StateVector containing analysis results

        """
        time_sel_matrix = self._calc_time_sel_matrix(obs_window_indices,
                                                     n_steps)
        if H is not None or h is None:
            return self._cycle_obsop(
                    xb.values, yo.values, yo.location_indices, yo.error_sd,
                    H, R, B, time_sel_matrix=time_sel_matrix, n_steps=n_steps)
        else:
            return self._cycle_obsop(
                    xb, yo, h, R, B, time_sel_matrix=time_sel_matrix, n_steps=n_steps)

    def step_forecast(self, xa, n_steps=1):
        if 'n_steps' in inspect.getfullargspec(self.model_obj.forecast).args:
            return self.model_obj.forecast(xa, n_steps=n_steps)
        else:
            if n_steps == 1:
                return self.model_obj.forecast(xa)
            else:
                out = [xa]
                xi = xa
                for s in range(n_steps):
                    xi = self.model.forecast(xi)
                    out.append(xi)
                return vector.StateVector(jnp.vstack(xi), store_as_jax=True)

    def _cycle_and_forecast(self, state_obs_tuple, filtered_idx):
        cur_state_vals = state_obs_tuple[0]
        obs_vals = state_obs_tuple[1]
        obs_times = state_obs_tuple[2]
        obs_loc_indices = state_obs_tuple[3]
        obs_error_sd = state_obs_tuple[4]

        cur_obs_vals = jax.lax.dynamic_slice_in_dim(obs_vals, filtered_idx[0],
                                                    len(filtered_idx))
        cur_obs_loc_indices = jax.lax.dynamic_slice_in_dim(obs_loc_indices,
                                                           filtered_idx[0],
                                                           len(filtered_idx))
        analysis, kh = self.step_cycle(
                vector.StateVector(values=cur_state_vals, store_as_jax=True),
                vector.ObsVector(values=cur_obs_vals,
                                 location_indices=cur_obs_loc_indices,
                                 error_sd=obs_error_sd,
                                 store_as_jax=True),
                n_steps=self.steps_per_window,
                obs_window_indices=self.obs_window_indices)

        return (analysis.values[-1], obs_vals, obs_times, obs_loc_indices,
                obs_error_sd), analysis.values[:-1]

    def cycle(self,
              input_state,
              start_time,
              obs_vector,
              obs_error_sd,
              timesteps,
              analysis_window,
              analysis_time_in_window=None):
        """Perform DA cycle repeatedly, including analysis and forecast

        Args:
            input_state (vector.StateVector): Input state.
            obs_vector (vector.ObsVector): Observations vector.
            obs_error_sd (float): Standard deviation of observation error.
                Typically not known, so provide a best-guess.
            start_time (float or datetime-like): Starting time.
            timesteps (int): Number of timesteps, in model time.
            analysis_window (float): Time window from which to gather
                observations for DA Cycle.
            analysis_time_in_window (float): Where within analysis_window
                to perform analysis. For example, 0.0 is the start of the
                window. Default is None, which selects the middle of the
                window.

        Returns:
            vector.StateVector of analyses and times.
        """

        if analysis_time_in_window is None:
            analysis_time_in_window = analysis_window/2

        # NOTE: This is clumsy, but it sets up jax.lax.scan, which is VERY fast
        # Working on a better way of setting it up
        all_times = (
                jnp.repeat(start_time + analysis_time_in_window, timesteps)
                + jnp.arange(0, timesteps*analysis_window,
                             analysis_window)
                     )
        all_filtered_idx = jnp.stack([jnp.where(
            # Greater than start of window
            (obs_vector.times > cur_time - analysis_window/2)
            # AND Less than end of window
            * (obs_vector.times < cur_time + analysis_window/2)
            # OR Equal to start of window end
            + jnp.isclose(obs_vector.times, cur_time - analysis_window/2,
                          rtol=0)
            # OR Equal to end of window
            + jnp.isclose(obs_vector.times, cur_time + analysis_window/2,
                          rtol=0)
            )[0] for cur_time in all_times])

        cur_state, all_values = jax.lax.scan(
                self._cycle_and_forecast,
                (input_state.values, obs_vector.values, obs_vector.times,
                 obs_vector.location_indices, obs_error_sd),
                all_filtered_idx)

        return vector.StateVector(values=jnp.vstack(all_values),
                                  store_as_jax=True)
