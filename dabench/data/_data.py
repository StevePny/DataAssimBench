"""Base class for data generator objects"""

import copy
import numpy as np
import jax.numpy as jnp
import xarray as xr
import warnings
from importlib import resources

from dabench.data._utils import integrate
from dabench import _suppl_data


class Data():
    """Generic class for data generator objects.

    Attributes:
        system_dim (int): system dimension
        time_dim (int): total time steps
        original_dim (tuple): dimensions in original space, e.g. could be 3x3
            for a 2d system with system_dim = 9. Defaults to (system_dim),
            i.e. 1d.
        random_seed (int): random seed, defaults to 37
        delta_t (float): the timestep of the data (assumed uniform)
        store_as_jax (bool): Store values as jax array instead of numpy array.
            Default is False (store as numpy).
        """

    def __init__(self,
                 system_dim=3,
                 time_dim=1,
                 original_dim=None,
                 random_seed=37,
                 delta_t=0.01,
                 store_as_jax=False,
                 x0=None,
                 **kwargs):
        """Initializes the base data object"""

        self.system_dim = system_dim
        self.time_dim = time_dim
        self.random_seed = random_seed
        self.delta_t = delta_t
        self.store_as_jax = store_as_jax
        # x0 attribute is property to better convert between jax/numpy
        self._x0 = x0

        if original_dim is None:
            self.original_dim = (system_dim,)
        else:
            self.original_dim = original_dim

        self._values_gridded = None
        self._x0_gridded = None

    @property
    def x0(self):
        return self._x0

    @x0.setter
    def x0(self, x0_vals):
        if x0_vals is None:
            self._x0 = None
        else:
            if self.store_as_jax:
                self._x0 = jnp.asarray(x0_vals)
            else:
                self._x0 = np.asarray(x0_vals)

    @x0.deleter
    def x0(self):
        del self._x0

    @property
    def x0_gridded(self):
        if self._x0 is None:
            return None
        else:
            return self._x0.reshape(self.original_dim)

    def generate(self, n_steps=None, t_final=None, x0=None, M0=None,
                 return_tlm=False, stride=None, **kwargs):
        """Generates a dataset and returns xarray state vector.

        Notes:
            Either provide n_steps or t_final in order to indicate the length
            of the forecast. These are used to set the values, times, and
            time_dim attributes.

        Args:
            n_steps (int): Number of timesteps. One of n_steps OR
                t_final must be specified.
            t_final (float): Final time of trajectory. One of n_steps OR
                t_final must be specified.
            M0 (ndarray): the initial condition of the TLM matrix computation
                shape (system_dim, system_dim).
            return_tlm (bool): specifies whether to compute and return the
                integrated Jacobian as a Tangent Linear Model for each
                timestep.
            x0 (ndarray): initial conditions state vector of shape (system_dim)
            stride (int): specify how many steps to skip in the output data
                versus the model timestep (delta_t).
            **kwargs: arguments to the integrate function (permits changes in
                convergence tolerance, etc.).

        Returns:
            Nothing if return_tlm=False. If return_tlm=True, a list
                of TLMs corresponding to the system trajectory.
        """

        # Check that n_steps or t_final is supplied
        if n_steps is not None:
            t_final = n_steps * self.delta_t
        elif t_final is not None:
            n_steps = int(t_final/self.delta_t)
        else:
            raise TypeError('Either n_steps or t_final must be supplied as an '
                            'input argument.')

        # Check that x0 initial conditions is supplied
        if x0 is None:
            if self.x0 is not None:
                x0 = self.x0
            else:
                raise TypeError('Initial condition is None, x0 = {}. it must '
                                'either be provided as an argument or set as '
                                'an attribute in the model object.'.format(x0))

        # Check that self.rhs is defined
        if self.rhs is None:
            raise AttributeError('self.rhs must be specified prior to '
                                 'calling generate.')

        # Set f matrix for integrate
        if return_tlm:
            # Prep x0
            if M0 is None:
                M0 = jnp.identity(self.system_dim)
            xaux0 = jnp.concatenate((x0.ravel(), M0.ravel()))
            x0 = xaux0
            # Check that self.rhs_aux is defined
            if self.rhs_aux is None:
                raise AttributeError('self.rhs_aux must be specified prior to '
                                     'calling generate.')
            f = self.rhs_aux
        else:
            f = self.rhs

        # Integrate and store values and times
        # If data object has its own integration method, use that
        if hasattr(self, 'integrate') and callable(getattr(self, 'integrate')):
            y, t = self.integrate(f, x0, t_final, self.delta_t, stride=stride,
                                  **kwargs)
        # Otherwise, use integrate from dabench.support.utils
        else:
            y, t = integrate(f, x0, t_final, self.delta_t, stride=stride,
                             jax_comps=self.store_as_jax,
                             **kwargs)

        # Convert to JAX if necessary
        if self.store_as_jax:
            y_out = jnp.array(y[:,:self.system_dim])
        else:
            y_out = np.array(y[:,:self.system_dim])
        # Build Xarray object for output
        out_vec = xr.Dataset(
            {'x': (['time','i'],y_out)},
            coords={'time': t,
                    'i':np.arange(self.system_dim)},
            attrs={'store_as_jax':self.store_as_jax,
                   'system_dim': self.system_dim,
                   'time_dim': self.time_dim
            }
        )

        # Return the data series and associated TLMs if requested
        if return_tlm:
            # Reshape M matrix
            if self.store_as_jax:
                M = jnp.reshape(y[:, self.system_dim:],
                                (self.time_dim,
                                self.system_dim,
                                self.system_dim)
                                )
            else:
                M = np.reshape(y[:, self.system_dim:],
                                (self.time_dim,
                                self.system_dim,
                                self.system_dim)
                                )
            return out_vec, M
        else:
            return out_vec

    def rhs_aux(self, x, t):
        """The auxiliary model used to compute the TLM.

        Args:
          x (ndarray): State vector with size (system_dim)
          t (ndarray): Array of times with size (time_dim)

        Returns:
          dxaux (ndarray): State vector [size: (system_dim,)]

        """
        # Compute M
        dxdt = self.rhs(x[:self.system_dim], t)
        J = self.Jacobian(x[:self.system_dim])
        M = jnp.array(jnp.reshape(x[self.system_dim:], (self.system_dim,
                                                        self.system_dim)))

        # Matrix multiplication
        dM = J @ M

        dxaux = jnp.concatenate((dxdt, dM.flatten()))

        return dxaux

    def calc_lyapunov_exponents_series(self, total_time=None, rescale_time=1,
                                       convergence=0.01, x0=None):
        """Computes the spectrum of Lyapunov Exponents.

        Notes:
            Lyapunov exponents help describe the degree of "chaos" in the
            model. Make sure to plot the output to check that the algorithm
            converges. There are three ways to make the estimate more accurate:
                1. Decrease the delta_t of the model
                2. Increase total_time
                3. Decrease rescale time (try increasing total_time first)
            Algorithm: Eckmann 85,
            https://www.ihes.fr/~ruelle/PUBLICATIONS/%5B81%5D.pdf pg 651
            This method computes the full time series of Lyapunov Exponents,
            which is useful for plotting for debugging. To get only the final
            Lyapunov Exponent, use self.calc_lyapunov_exponents.

        Args:
            total_time (float) : Time to integrate over to compute LEs.
                Usually there's a tradeoff between accuracy and computation
                time (more total_time leads to higher accuracy but more
                computation time). Default depends on model type and are based
                roughly on how long it takes for satisfactory convergence:
                For Lorenz63: n_steps=15000 (total_time=150 for delta_t=0.01)
                For Lorenz96: n_steps=50000 (total_time=500 for delta_t=0.01)
            rescale_time (float) : Time for when the algorithm rescales the
                propagator to reduce the exponential growth in errors.
                Default is 1 (i.e. 100 timesteps when delta_t = 0.01).
            convergence (float) : Prints warning if LE convergence is below
                this number. Default is 0.01.
            x0 (array) : initial condition to start computing LE.  Needs
                to be on the attractor (i.e., remove transients). Default is
                None, which will fallback to use the x0 set during model object
                initialization.

        Returns:
            Lyapunov exponents for all timesteps, array of size
                (total_time/rescale_time - 1, system_dim)
        """

        # Set total_time
        if total_time is None:
            subclass_name = self.__class__.__name__
            if subclass_name == 'Lorenz63':
                total_time = int(15000*self.delta_t)
            elif subclass_name == 'Lorenz96':
                total_time = int(50000*self.delta_t)
            else:
                total_time = 100

        if rescale_time > total_time:
            raise ValueError('rescale_time must be less than or equal to '
                             'total_time. Current values are rescale_time = {}'
                             ' and total_time = {}'.format(rescale_time,
                                                           total_time))
        D = self.system_dim
        times = jnp.arange(0, total_time, rescale_time)

        # Array to be populated with Lyapunov Exponents
        LE = jnp.zeros((len(times)-1, D))

        # Set initial conditions for first time period
        M0 = jnp.eye(D)
        prev_R = jnp.zeros(D)

        # Loop over rescale time periods
        for i, (t1, t2) in enumerate(zip(times[:-1], times[1:])):

            x, M = self.generate(t_final=t2-t1, x0=x0, M0=M0, return_tlm=True)
            x_t2 = x.isel(time=-1).to_array().data.flatten()
            M_t2 = M[-1]

            Q, R = jnp.linalg.qr(M_t2)

            curr_R = jnp.log(jnp.abs(jnp.diag(R)))
            LE = LE.at[i].set((prev_R+curr_R)/t2)
            prev_R += curr_R

            x0 = x_t2
            M0 = Q
        # Check convergence
        compute_conv = jnp.mean(jnp.abs(LE[-11:-1] - LE[-10:]), axis=0)
        if jnp.any(compute_conv > convergence):
            print('WARNING: Exceeding convergence = {} > {}. Increase total_'
                  'time and/or decrease rescale_time.'.format(compute_conv,
                                                              convergence))

        return LE

    def calc_lyapunov_exponents_final(self, total_time=None, rescale_time=1,
                                      convergence=0.05, x0=None):
        """Computes the final Lyapunov Exponents

        Notes:
            See self.calc_lyapunov_exponents_series for full info

        Args:
            total_time (float) : Time to integrate over to compute LEs.
                Usually there's a tradeoff between accuracy and computation
                time (more total_time leads to higher accuracy but more
                computation time). Default depends on model type and are based
                roughly on how long it takes for satisfactory convergence:
                For Lorenz63: n_steps=15000 (total_time=150 for delta_t=0.01)
                For Lorenz96: n_steps=50000 (total_time=500 for delta_t=0.01)
            rescale_time (float) : Time for when the algorithm rescales the
                propagator to reduce the exponential growth in errors.
                Default is 1 (i.e. 100 timesteps when delta_t = 0.01).
            convergence (float) : Prints warning if LE convergence is below
                this number. Default is 0.01.
            x0 (array) : initial condition to start computing LE.  Needs
                to be on the attractor (i.e., remove transients). Default is
                None, which will fallback to use the x0 set during model object
                initialization.

        Returns:
            Lyapunov exponents array of size (system_dim)
        """

        return self.calc_lyapunov_exponents_series(total_time=total_time,
                                                   rescale_time=rescale_time,
                                                   x0=x0,
                                                   convergence=convergence)[-1]
