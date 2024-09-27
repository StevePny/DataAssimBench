"""Class for Var 4D Backpropagation Data Assimilation Cycler object"""

import inspect
import warnings

import numpy as np
import jax.numpy as jnp
import jax.scipy as jscipy
from jax import grad, value_and_grad
from jax.scipy import optimize
import jax
import optax
from functools import partial
import xarray as xr
import xarray_jax as xj

from dabench import dacycler, vector
import dabench.dacycler._utils as dac_utils


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
        num_iters (int): Number of iterations for backpropagation per analysis
            cycle. Default is 3.
        steps_per_window (int): Number of timesteps per analysis window.
            If None (default), will calculate automatically based on delta_t
            and .cycle() analysis_window length.
        learning_rate (float): LR for backpropogation. Default is 0.5, but
            DA results can be quite sensitive to this parameter.
        lr_decay (float): Exponential learning rate decay. If set to 1,
            no decay. Default is 0.5.
        obs_window_indices (list): Timestep indices where observations fall
            within each analysis window. For example, if analysis window is
            0 - 0.05 with delta_t = 0.01 and observations fall at 0, 0.01,
            0.02, 0.03, 0.04, and 0.05, obs_window_indices =
            [0, 1, 2, 3, 4, 5]. If None (default), will calculate
            automatically.
        loss_growth_limit (float): If loss grows by more than this factor
            during one analysis cycle, JAX will cut off computation and
            return an error. This prevents it from hanging indefinitely
            when loss grows exponentionally. Default is 10.
    """

    def __init__(self,
                 system_dim=None,
                 delta_t=None,
                 model_obj=None,
                 B=None,
                 R=None,
                 H=None,
                 h=None,
                 learning_rate=0.5,
                 lr_decay=0.5,
                 num_iters=3,
                 steps_per_window=None,
                 obs_window_indices=None,
                 loss_growth_limit=10,
                 **kwargs
                 ):

        self.num_iters = num_iters
        self.learning_rate = learning_rate
        self.lr_decay = lr_decay
        self.steps_per_window = steps_per_window
        self.obs_window_indices = obs_window_indices
        self.loss_growth_limit = loss_growth_limit

        # Var4D Backprop requires H to be a JAX array
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

    def _raise_nan_error(self):
        raise ValueError('Loss value is nan, exiting optimization')
        
    def _raise_loss_growth_error(self):
        raise ValueError('Loss value has exceeded self.loss_growth_limit, exiting optimization')

    def _callback_raise_error(self, error_method, loss_val):
        jax.debug.callback(error_method)
        return loss_val

    @partial(jax.jit, static_argnums=[0])
    def _calc_obs_term(self, pred_x, obs_vals, Ht, Rinv):
        pred_obs = pred_x @ Ht
        resid = pred_obs.ravel() - obs_vals.ravel()

        return jnp.sum(resid.T @ Rinv @ resid)

    def _make_loss(self, xb0, obs_vals,  Hs, Binv, Rinv,
                   obs_window_indices,
                   obs_time_mask, n_steps):
        """Define loss function based on 4dvar cost"""

        @jax.jit
        def loss_4dvarcost(x0):
            # Get initial departure
            db0 = (x0.to_array().data.ravel() - xb0.to_array().data.ravel())

            # Make new prediction
            # NOTE: [1] selects the full forecast instead of last timestep only
            pred_x = self._step_forecast(
                x0, n_steps)[1].to_stacked_array('system',['time']).data

            # Calculate observation term of J_0
            obs_term = 0
            for i, j in enumerate(obs_window_indices):
                obs_term += jax.lax.cond(
                        obs_time_mask.at[i].get(mode='fill', fill_value=0),
                        lambda: self._calc_obs_term(pred_x[j], obs_vals[i],
                                                    Hs.at[i].get(mode='clip').T,
                                                    Rinv),
                        lambda: 0.0
                        )

            # Calculate initial departure term of J_0 based on original x0
            initial_term = (db0.T @ Binv @ db0)

            # Cost is the sum of the two terms
            loss_val = initial_term + obs_term
            return jax.lax.cond(
                    jnp.isnan(loss_val),
                    lambda: self._callback_raise_error(self._raise_nan_error,
                                                       loss_val),
                    lambda: loss_val)

        return loss_4dvarcost

    def _make_backprop_epoch(self, loss_func, optimizer, hessian_inv):

        loss_value_grad = value_and_grad(loss_func, argnums=0)

        @jax.jit
        def _backprop_epoch(epoch_state_tuple, i):
            x0, init_loss, opt_state = epoch_state_tuple
            x0 = x0.to_xarray()
            loss_val, dx0 = loss_value_grad(x0)
            x0_array = x0.to_stacked_array('system', [])
            dx0_hess = hessian_inv @ dx0.to_stacked_array('system',[]).data
            init_loss = jax.lax.cond(
                    i == 0,
                    lambda: loss_val,
                    lambda: init_loss)
            loss_val = jax.lax.cond(
                    loss_val/init_loss > self.loss_growth_limit,
                    lambda: self._callback_raise_error(
                        self._raise_loss_growth_error, loss_val),
                    lambda: loss_val)

            updates, opt_state = optimizer.update(dx0_hess, opt_state)
            x0_array.data = optax.apply_updates(
                x0_array.data, updates)
            x0_new = x0_array.to_unstacked_dataset('system').assign_attrs(
                x0.attrs
            )
            return (xj.from_xarray(x0_new), init_loss, opt_state), loss_val

        return _backprop_epoch

    def _cycle_obsop(self, x0_xarray, obs_values, obs_loc_indices,
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

        Rinv = jscipy.linalg.inv(R)
        Binv = jscipy.linalg.inv(B)

        # Compute Hessian
        hessian_inv = jscipy.linalg.inv(
                Binv + Hs.at[0].get().T @ Rinv @ Hs.at[0].get())

        loss_func = self._make_loss(
                x0_xarray,
                obs_values,
                Hs,
                Binv,
                Rinv,
                obs_window_indices,
                obs_time_mask,
                n_steps=self.steps_per_window)

        lr = optax.exponential_decay(
                self.learning_rate,
                1,
                self.lr_decay)
        optimizer = optax.sgd(lr)
        opt_state = optimizer.init(x0_xarray.to_stacked_array('system',[]).data)

        # Make initial forecast and calculate loss
        backprop_epoch_func = self._make_backprop_epoch(loss_func, optimizer,
                                                        hessian_inv)
        epoch_state_tuple, loss_vals = jax.lax.scan(
                backprop_epoch_func, init=(xj.from_xarray(x0_xarray), 0., opt_state),
                xs=jnp.arange(self.num_iters))

        x0_new = epoch_state_tuple[0].to_xarray()

        return x0_new