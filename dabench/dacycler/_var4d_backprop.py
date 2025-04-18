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
from typing import Callable, Any

from dabench import dacycler
import dabench.dacycler._utils as dac_utils
from dabench.model import Model

# For typing
ArrayLike = np.ndarray | jax.Array
XarrayDatasetLike = xr.Dataset | xj.XjDataset
ScheduleState = Any

class Var4DBackprop(dacycler.DACycler):
    """Class for building Backpropagation 4D DA Cycler

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
        num_iters: Number of iterations for backpropagation per analysis
            cycle. Default is 3.
        steps_per_window: Number of timesteps per analysis window.
            If None (default), will calculate automatically based on delta_t
            and .cycle() analysis_window length.
        learning_rate: LR for backpropogation. Default is 0.5, but
            DA results can be quite sensitive to this parameter.
        lr_decay: Exponential learning rate decay. If set to 1,
            no decay. Default is 0.5.
        obs_window_indices: Timestep indices where observations fall
            within each analysis window. For example, if analysis window is
            0 - 0.05 with delta_t = 0.01 and observations fall at 0, 0.01,
            0.02, 0.03, 0.04, and 0.05, obs_window_indices =
            [0, 1, 2, 3, 4, 5]. If None (default), will calculate
            automatically.
        loss_growth_limit: If loss grows by more than this factor
            during one analysis cycle, JAX will cut off computation and
            return an error. This prevents it from hanging indefinitely
            when loss grows exponentionally. Default is 10.
    """
    in_4d = True
    ensemble = False

    def __init__(self,
                 system_dim: int,
                 delta_t: float,
                 model_obj: Model,
                 B: ArrayLike | None = None,
                 R: ArrayLike | None = None,
                 H: ArrayLike | None = None,
                 h: Callable | None = None,
                 learning_rate: float = 0.5,
                 lr_decay: float = 0.5,
                 num_iters: int = 3,
                 steps_per_window: int | None = None,
                 obs_window_indices: ArrayLike | list | None = None,
                 loss_growth_limit: float = 10,
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

    def _raise_nan_error(self):
        raise ValueError('Loss value is nan, exiting optimization')
        
    def _raise_loss_growth_error(self):
        raise ValueError('Loss value has exceeded self.loss_growth_limit, exiting optimization')

    def _callback_raise_error(self,
                              error_method: Callable,
                              loss_val: float
                              ) -> float:
        jax.debug.callback(error_method)
        return loss_val

    # @partial(jax.jit, static_argnums=[0])
    def _calc_obs_term(self,
                       X: ArrayLike,
                       obs_vals: ArrayLike,
                       Ht: ArrayLike,
                       Rinv: ArrayLike
                       ) -> jax.Array:
        Y = X @ Ht
        resid = Y.ravel() - obs_vals.ravel()

        return jnp.sum(resid.T @ Rinv @ resid)

    def _make_loss(self,
                   xb0: XarrayDatasetLike,
                   obs_vals: ArrayLike,
                   Hs: ArrayLike,
                   Binv: ArrayLike,
                   Rinv: ArrayLike,
                   obs_window_indices: ArrayLike | list,
                   obs_time_mask: ArrayLike,
                   n_steps: int
                   ) -> Callable:
        """Define loss function based on 4dvar cost"""

        # @jax.jit
        def loss_4dvarcost(x0: XarrayDatasetLike) -> jax.Array:
            # Get initial departure
            db0 = (x0.to_array().data.ravel() - xb0.to_array().data.ravel())

            # Make new prediction
            # NOTE: [1] selects the full forecast instead of last timestep only
            X = self._step_forecast(
                x0, n_steps)[1].to_stacked_array('system',['time']).data

            # Calculate observation term of J_0
            obs_term = 0
            for i, j in enumerate(obs_window_indices):
                obs_term += jax.lax.cond(
                        obs_time_mask.at[i].get(mode='fill', fill_value=0),
                        lambda: self._calc_obs_term(X[j], obs_vals[i],
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

    def _make_backprop_epoch(self,
                             loss_func: Callable,
                             optimizer: optax.GradientTransformation,
                             hessian_inv: ArrayLike):

        loss_value_grad = value_and_grad(loss_func, argnums=0)

        # @jax.jit
        def _backprop_epoch(
                epoch_state_tuple: tuple[XarrayDatasetLike, ArrayLike, ScheduleState],
                i: int
                ) -> tuple[tuple[XarrayDatasetLike, ArrayLike, ScheduleState], ArrayLike]:
            x0_ds, init_loss, opt_state = epoch_state_tuple
            x0_ds = x0_ds.to_xarray()
            loss_val, dx0 = loss_value_grad(x0_ds)
            x0_ar = x0_ds.to_stacked_array('system', [])
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
            x0_ar.data = optax.apply_updates(
                x0_ar.data, updates)
            xa0_ds = x0_ar.to_unstacked_dataset('system').assign_attrs(
                x0_ds.attrs
            )
            return (xj.from_xarray(xa0_ds), init_loss, opt_state), loss_val

        return _backprop_epoch

    def _cycle_obsop(self,
                     xb0_ds: XarrayDatasetLike,
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

        Rinv = jscipy.linalg.inv(R)
        Binv = jscipy.linalg.inv(B)

        # Compute Hessian
        hessian_inv = jscipy.linalg.inv(
                Binv + Hs.at[0].get().T @ Rinv @ Hs.at[0].get())

        loss_func = self._make_loss(
                xb0_ds,
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
        opt_state = optimizer.init(xb0_ds.to_stacked_array('system',[]).data)

        # Make initial forecast and calculate loss
        backprop_epoch_func = self._make_backprop_epoch(loss_func, optimizer,
                                                        hessian_inv)
        epoch_state_tuple, loss_vals = jax.lax.scan(
                backprop_epoch_func, init=(xj.from_xarray(xb0_ds), 0., opt_state),
                xs=jnp.arange(self.num_iters))

        xa0_ds = epoch_state_tuple[0].to_xarray()

        return xa0_ds
