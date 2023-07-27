"""Class for 4D Backpropagation  Data Assimilation Cycler object"""

import inspect

import numpy as np
import jax.numpy as jnp
import jax.scipy as jscipy
from jax import value_and_grad

from dabench import dacycler, vector


class Backprop4D(dacycler.DACycler):
    """Class for building Backpropagation 4D DA Cycler"""

    def __init__(self,
                 system_dim=None,
                 delta_t=None,
                 ensemble=False,
                 model_obj=None,
                 B=None,
                 R=None,
                 h=None,
                 H=None,
                 learning_rate=1e-5,
                 num_epochs=20,
                 **kwargs
                 ):

        self.h = h
        self.H = H
        self.R = R
        self.B = B
        self.num_epochs = num_epochs
        self.learning_rate = learning_rate

        super().__init__(system_dim=system_dim,
                         delta_t=delta_t,
                         model_obj=model_obj)

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
            return self._cycle_linear_obsop(xb, yo, H, R, B, n_steps=n_steps,
                                            time_sel_matrix=time_sel_matrix)
        else:
            return self._cycle_general_obsop(xb, yo, h, R, B, n_steps=n_steps,
                                             time_sel_matrix=time_sel_matrix)

    def _calc_default_H(self, obs_vec):
        """If H is not provided, creates identity matrix to serve as H"""
        H = jnp.zeros((obs_vec.values.flatten().shape[0], self.system_dim))
        H = H.at[jnp.arange(H.shape[0]), obs_vec.location_indices.flatten()
                 ].set(1)
        return H

    def _calc_default_R(self, obs_vec):
        """If R i s not provided, calculates default based on observation error"""
        return jnp.identity(obs_vec.values.flatten().shape[0])*obs_vec.error_sd**2

    def _calc_default_B(self):
        """If B is not provided, identity matrix with shape (system_dim, system_dim."""

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

    def _cycle_general_obsop(self, forecast, obs_vec):
        return

    def _cycle_linear_obsop(self, forecast, obs_vec, H=None, R=None,
                            B=None, time_sel_matrix=None, n_steps=1):
        """When obsop (H) is linear"""
        if H is None:
            if self.H is None:
                H = self._calc_default_H(obs_vec)
            else:
                H = self.H
        if R is None:
            if self.R is None:
                R = self._calc_default_R(obs_vec)
            else:
                R = self.R
        if B is None:
            if self.B is None:
                B = self._calc_default_B()
            else:
                B = self.B

        r0 = forecast.values
        loss_func = self._make_loss(obs_vec.values, H, B, R, time_sel_matrix,
                                    rb0=r0, n_steps=n_steps)
        for e in range(self.num_epochs):
            epoch_loss, dr0 = value_and_grad(loss_func, argnums=0)(r0)
            print(e, epoch_loss)
            r0 -= self.learning_rate*dr0

        ra = self.step_forecast(
                vector.StateVector(values=r0, store_as_jax=True),
                n_steps=n_steps)

        return ra, None

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

    def cycle(self,
              input_state,
              start_time,
              obs_vector,
              timesteps,
              analysis_window,
              obs_window_indices=[0],
              steps_per_window=1,
              analysis_time_in_window=None):
        """Perform DA cycle repeatedly, including analysis and forecast

        Args:
            input_state (vector.StateVector): Input state.
            obs_vector (vector.ObsVector): Observations vector.
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
        # For storing outputs
        all_analyses = []
        all_times = []
        cur_time = start_time + analysis_time_in_window
        cur_state = input_state

        for i in range(timesteps):
            # 1. Filter observations to plus/minus 0.1 from that time
            obs_vec_timefilt = obs_vector.filter_times(
                cur_time - analysis_window/2, cur_time + analysis_window/2)

            if obs_vec_timefilt.values.shape[0] > 0:
                # 2. Calculate analysis
                analysis, kh = self.step_cycle(
                        cur_state, obs_vec_timefilt,
                        n_steps=steps_per_window,
                        obs_window_indices=obs_window_indices)
                # 3. Save outputs
                print(analysis.values.shape)
                all_analyses.append(analysis.values[:-1])
                all_times.append(cur_time)

            cur_time += analysis_window
            cur_state = analysis[-1]

        return vector.StateVector(values=jnp.vstack(all_analyses),
                                  store_as_jax=True)
