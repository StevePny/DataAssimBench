"""Class for Var 4D Backpropagation Data Assimilation Cycler object"""

import inspect

import numpy as np
import jax.numpy as jnp
import jax.scipy as jscipy
from jax import grad, value_and_grad
from jax.scipy import optimize
import jax
import optax

from dabench import dacycler, vector


class Var4DBackprop(dacycler.DACycler):
    """Class for building Backpropagation 4D DA Cycler

    Attributes:
        system_dim (int): System dimension.
        delta_t (float): The timestep of the model (assumed uniform)
        model_obj (dabench.Model): Forecast model object.
        in_4d (bool): True for 4D data assimilation techniques (e.g. 4DVar).
            Always True for Var4DBackprop.
        ensemble (bool): True for ensemble-based data assimilation techniques
            (ETKF). Always False for Var4DBackprop.
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
        num_epochs (int): Number of epochs for backpropagation per analysis
            cycle. Default is 20.
        steps_per_window (int): Number of timesteps per analysis window.
        learning_rate (float): LR for backpropogation. Default is 1e-5, but
            DA results can be quite sensitive to this parameter.
        lr_decay (float): Exponential learning rate decay. If set to 1,
            no decay. Default is 1.
        obs_window_indices (list): Timestep indices where observations fall
            within each analysis window. For example, if analysis window is
            0 - 0.05 with delta_t = 0.01 and observations fall at 0, 0.01,
            0.02, 0.03, 0.04, and 0.05, obs_window_indices =
            [0, 1, 2, 3, 4, 5].
    """

    def __init__(self,
                 system_dim=None,
                 delta_t=None,
                 model_obj=None,
                 B=None,
                 R=None,
                 H=None,
                 h=None,
                 learning_rate=1e-5,
                 lr_decay=1.0,
                 num_epochs=20,
                 steps_per_window=1,
                 obs_window_indices=[0],
                 **kwargs
                 ):

        self.num_epochs = num_epochs
        self.learning_rate = learning_rate
        self.lr_decay = lr_decay
        self.steps_per_window = steps_per_window
        self.obs_window_indices = obs_window_indices

        super().__init__(system_dim=system_dim,
                         delta_t=delta_t,
                         model_obj=model_obj,
                         in_4d=True,
                         ensemble=False,
                         B=B, R=R, H=H, h=h)

    def _calc_default_H(self, obs_values, obs_loc_indices):
        H = jnp.zeros((obs_values.flatten().shape[0], self.system_dim))
        H = H.at[jnp.arange(H.shape[0]), obs_loc_indices.flatten()
                 ].set(1)
        return H

    def _calc_default_R(self, obs_values, obs_error_sd):
        return jnp.identity(obs_values.flatten().shape[0])*(obs_error_sd**2)

    def _calc_default_B(self):
        return jnp.identity(self.system_dim)

    def _make_loss(self, obs_vals, H, B, R, time_sel_matrix, n_steps):
        """Define loss function based on 4dvar cost"""
        Rinv = jscipy.linalg.inv(R)
        Binv = jscipy.linalg.inv(B)

        @jax.jit
        def loss_4dvarcost(x0):
            pred_x = self.step_forecast(
                    vector.StateVector(values=x0, store_as_jax=True),
                    n_steps=n_steps).values

            # Make prediction based on current r
            xb0 = pred_x[0]

            # Apply observation operator to map to obs spcae
            pred_obs = time_sel_matrix @ pred_x @ H

            # Calculate observation term of J_0
            resid = (pred_obs.ravel() - obs_vals.ravel())
            obs_term = 0.5*np.sum(resid.T @ Rinv @ resid)

            # Calculate initial departure term of J_0 based on original x0
            db0 = (x0.ravel() - xb0.ravel())
            initial_term = 0.5*(db0.T @ Binv @ db0)

            # Cost is the sum of the two terms
            return initial_term + obs_term

        return loss_4dvarcost

    def _calc_time_sel_matrix(self, obs_steps_inds, n_pred_steps):
        time_sel_matrix = jnp.zeros((len(obs_steps_inds), n_pred_steps))
        time_sel_matrix = time_sel_matrix.at[
                jnp.arange(time_sel_matrix.shape[0]), obs_steps_inds].set(1)
        return time_sel_matrix

    def _make_backprop_epoch(self, optimizer):

        @jax.jit
        def _backprop_epoch(x0_opt_state_tuple, i):
            x0, dx0, i, opt_state = x0_opt_state_tuple
            updates, opt_state = optimizer.update(dx0, opt_state)
            x0_new = optax.apply_updates(x0, updates)

            return (x0_new, dx0, i+1, opt_state), x0_new

        return _backprop_epoch

    def _cycle_obsop(self, xb, obs_values, obs_loc_indices, obs_error_sd,
                     H=None, h=None, R=None, B=None, time_sel_matrix=None,
                     n_steps=1):
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

        x0 = xb
        loss_func = self._make_loss(obs_values, H, B, R, time_sel_matrix,
                                    n_steps=n_steps)

        lr = optax.exponential_decay(
                self.learning_rate,
                self.num_epochs, self.lr_decay)
        optimizer = optax.adam(lr)
        opt_state = optimizer.init(x0)

        # Make initial forecast and calculate loss
        loss_val, dx0 = value_and_grad(loss_func, argnums=0)(x0)
        backprop_epoch_func = self._make_backprop_epoch(optimizer)
        x0_opt_state_tuple, x0_vals = jax.lax.scan(
                backprop_epoch_func, init=(x0, dx0, 0, opt_state),
                xs=None, length=self.num_epochs)

        x0, dx0, i, opt_state = x0_opt_state_tuple

        # Analysis
        loss_val_end, dx0 = value_and_grad(loss_func, argnums=0)(x0)
        xa = self.step_forecast(
                vector.StateVector(values=x0, store_as_jax=True),
                n_steps=n_steps)

        return xa, jnp.array([loss_val, loss_val_end])

    def step_cycle(self, xb, yo, H=None, h=None, R=None, B=None, n_steps=1,
                   obs_window_indices=[0]):
        """Perform one step of DA Cycle"""
        time_sel_matrix = self._calc_time_sel_matrix(obs_window_indices,
                                                     n_steps)
        if H is not None or h is None:
            return self._cycle_obsop(
                    xb.values, yo.values, yo.location_indices, yo.error_sd,
                    H, R, B, time_sel_matrix=time_sel_matrix, n_steps=n_steps)
        else:
            return self._cycle_obsop(
                    xb, yo, h, R, B, time_sel_matrix=time_sel_matrix,
                    n_steps=n_steps)

    def step_forecast(self, xa, n_steps=1):
        """Perform forecast using model object"""
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

    def _cycle_and_forecast(self, cur_state_vals, filtered_idx):
        obs_vals = self._obs_vector.values
        obs_loc_indices = self._obs_vector.location_indices
        obs_error_sd = self._obs_error_sd

        cur_obs_vals = jax.lax.dynamic_slice_in_dim(obs_vals, filtered_idx[0],
                                                    len(filtered_idx))
        cur_obs_loc_indices = jax.lax.dynamic_slice_in_dim(obs_loc_indices,
                                                           filtered_idx[0],
                                                           len(filtered_idx))
        analysis, loss_vals = self.step_cycle(
                vector.StateVector(values=cur_state_vals, store_as_jax=True),
                vector.ObsVector(values=cur_obs_vals,
                                 location_indices=cur_obs_loc_indices,
                                 error_sd=obs_error_sd,
                                 store_as_jax=True),
                n_steps=self.steps_per_window,
                obs_window_indices=self.obs_window_indices)

        return analysis.values[-1], (analysis.values[:-1], loss_vals)

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

        # Set up for jax.lax.scan, which is very fast
        all_times = (
                jnp.repeat(start_time + analysis_time_in_window, timesteps)
                + jnp.arange(0, timesteps*analysis_window,
                             analysis_window)
                     )
        # Get the obs vectors for each analysis window
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

        self._obs_vector = obs_vector
        self._obs_error_sd = obs_error_sd
        cur_state, all_values = jax.lax.scan(
                self._cycle_and_forecast,
                init=input_state.values,
                xs=all_filtered_idx)
        all_losses = all_values[1]
        print(all_losses[:, -3:])
        all_values = all_values[0]

        return vector.StateVector(
                values=jnp.vstack(all_values),
                store_as_jax=True)
