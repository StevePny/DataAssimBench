"""Class for reservoir computing model for NWP"""

import logging
from copy import deepcopy
import pickle

from scipy import sparse, stats, linalg
import numpy as np
import jax
import jax.numpy as jnp
import xarray as xr

from dabench import vector, model

logging.basicConfig(filename='logfile.log', level=logging.DEBUG)

# For typing
ArrayLike = np.ndarray | jax.Array


class RCModel(model.Model):
    """A simple Reservoir Computing data-driven model

    Args:
        system_dim: Dimension of reservoir output.
        input_dim: Dimension of reservoir input signal.
        reservoir_dim: Dimension of reservoir state. Default: 512.
        sparsity: the percentage of zero-valued entries in the
            adjacency matrix (A). Default: 0.99.
        sparse_adj_matrix: If True, A is computed using scipy sparse.
        sigma: Scaling of the input weight matrix. Default: 0.5.
        sigma_bias: Bias term for sigma. Default: 0.
        leak_rate: ``(1-leak_rate)`` of the reservoir state at the
            previous time step is incorporated during timestep update.
            Default: 1.
        spectral_radius: Scaling of the reservoir adjacency
            matrix. Default: 0.9.
        tikhonov_parameter: Regularization parameter in the linear
            solving, penalizing amplitude of weight matrix elements. Default: 0.0.
        readout_method: How to handle reservoir state elements during
            readout. One of 'linear', 'biased', or 'quadratic'.
            Default: 'linear'.
        random_seed: Random seed for random number generation. Default
            is 1.

    Attributes:
        s (ArrayLike): Model states over entire time series.
        s_last (ArrayLike): Last model state
        ybar (ArrayLike): y.T @ st, set in _compute_Wout.
        sbar (ArrayLike): st.T @ st, set in _compute_Wout.
        A (ArrayLike): reservoir adjacency weight matrix, set in
            ``.weights_init()``.
        Win (ArrayLike): reservoir input weight matrix, set in
            ``.weights_init()``.
        Wout (ArrayLike): trained output weight matrix, set in ``.train()``.
    """

    def __init__(self,
                 system_dim: int,
                 input_dim: int,
                 reservoir_dim:int = 512,
                 sparsity: float = 0.99,
                 sparse_adj_matrix: bool = False,
                 sigma: float = 0.5,
                 sigma_bias: float = 0.,
                 leak_rate: float = 1.0,
                 spectral_radius: float = 0.9,
                 tikhonov_parameter: float = 0.,
                 readout_method: bool = 'linear',
                 random_seed: int = 1,
                 **kwargs):

        self.system_dim = system_dim
        self.input_dim = input_dim
        self.reservoir_dim = reservoir_dim

        self.sparsity = sparsity
        self.sparse_adj_matrix = sparse_adj_matrix
        self.sigma = sigma
        self.sigma_bias = sigma_bias
        self.spectral_radius = spectral_radius
        self.leak_rate = leak_rate

        if readout_method not in ['linear', 'biased', 'quadratic']:
            raise ValueError(
                'readout_method must be one of: "linear", "biased", or '
                '"quadratic". \n Got {} instead. Default is "linear".'.format(
                    readout_method))
        self.readout_method = readout_method
        self.tikhonov_parameter = tikhonov_parameter

        self.ybar = 0
        self.sbar = 0

        self._random_num_generator = np.random.default_rng(random_seed)

    def weights_init(self):
        """Initialize the weight matrices

        Notes:
            Generate the random adjacency (A) and input weight matrices (Win)
            with sparsity determined by ``sparsity`` attribute,
            scaled by ``spectral_radius`` and ``sigma`` parameters,
            respectively. If sparse_adj_matrix is True, then use scipy sparse
            matrices for computational speed up.

        Sets Attributes:
            A (ArrayLike): (reservoir_dim, reservoir_dim),
                reservoir adjacency matrix
            Win (ArrayLike): (reservoir_dim, input_dim),
                reservoir input weight matrix
            Adense (ArrayLike): stores dense version of A if A is specified
                as sparse format.
        """

        # Create adjacency weight matrix that defines reservoir dynamics
        # Dense version
        if not self.sparse_adj_matrix:
            # Initialize weights with a random matrix centered around zero:
            A = (self._random_num_generator.random(
                    (self.reservoir_dim, self.reservoir_dim))
                 - 0.5)

            # Delete the fraction of connections given by (self.sparsity):
            A[self._random_num_generator.random(A.shape) < self.sparsity] = 0

            # Compute the spectral radius of self.Win
            radius = np.max(np.abs(np.linalg.eigvals(A)))

            # Rescale the adjacencey matrix to the requested spectral radius
            A = A * (self.spectral_radius / radius)

        # Sparse version
        else:
            # Uniform dsit between [loc, loc+scale]
            uniform = stats.uniform(-1.0, 2.)
            uniform.random_state = self._random_num_generator
            A = sparse.random(self.reservoir_dim, self.reservoir_dim,
                              density=(1-self.sparsity), format='csr',
                              data_rvs=uniform.rvs,
                              random_state=self._random_num_generator)

            try:
                eig = sparse.linalg.eigs(A, k=1, return_eigenvectors=False,
                                         which='LM', tol=1e-10)
            except sparse.linalg.ArpackNoConvergence as err:
                k = len(err.eigenvalues)
                if k <= 0:
                    raise AssertionError(
                            "Spurious no-eigenvalues-found case") from err
                eig = err.eigenvalues

            radius = np.max(np.abs(eig))
            A.data = A.data * (self.spectral_radius / radius)

        if self.sigma_bias == 0:
            # random input weights:
            Win = (self._random_num_generator.random(
                (self.reservoir_dim, self.input_dim)) * 2 - 1)
            Win = self.sigma * Win
        else:
            Win = (self._random_num_generator.random(
                (self.reservoir_dim, self.input_dim)) * 2 - 1)
            Win_input = np.ones((self.reservoir_dim, 1))
            Win = self.sigma * Win
            Win_input *= self.sigma_bias
            Win = np.hstack([Win_input, Win])

        self.A = A
        self.Win = Win
        self.states = None
        self.Adense = A.asformat('array') if self.sparse_adj_matrix else A

    def readin(self,
               state_vec: xr.Dataset,
               A: ArrayLike | None = None,
               Win: ArrayLike | None = None,
               r0: ArrayLike | None = None
               ) -> xr.Dataset:
        """Generate reservoir state time series from input time series

        Args:
            state_vec: input signal to reservoir, size (time_dim, input_dim)
            A: Reservoir adjacency martrix, size (reservoir_dim,
                reservoir_dim). If None, will use self.A.
            Win: Reservoir input weight matrix, size (reservoir_dim,
                input_dim). If None, will use self.Win.
            r0: Initial reservoir state, size (reservoir_dim,). If None,
                start from all 0s.

        Returns:
            Reservoirs state, size (time_dim, reservoir_dim)
        """
        u = state_vec.to_stacked_array('system',['time']).data
        r = np.zeros((u.shape[0], self.reservoir_dim))

        if r0 is not None:
            logging.debug(
                    'readin:: using initial reservoir state: %s', r0)
            r[0, :] = np.reshape(r0, (1, self.reservoir_dim))

        # Encoding input signal {u(t)} -> {s(t)}
        for t in range(0, u.shape[0]):
            r[t, :] = self.update(r[t - 1], u[t - 1, :], A, Win)

        return xr.Dataset(
            {'r': (('time', 'reservoir'), r)},
            coords={'time':state_vec.time}
        )

    def update(self,
               r: ArrayLike,
               u: ArrayLike,
               A: ArrayLike | None = None,
               Win: ArrayLike | None = None
               ) -> ArrayLike:
        """Update reservoir state with input signal and previous state

        Args:
            r: Previous reservoir state, size (reservoir_dim,).
            u: input signal, size (input_dim,) 
            A: Reservoir adjacency martrix, size (reservoir_dim,
                reservoir_dim). If None, will use self.A.
            Win: Reservoir input weight matrix, size (reservoir_dim,
                input_dim). If None, will use self.Win.

        Returns:
            Reservoir state at next time step, size (reservoir_dim,)
        """

        if A is None:
            A = self.A
        if Win is None:
            Win = self.Win

        try:
            if self.sigma_bias != 0:
                u = jnp.concatenate((jnp.array([1.0]), u))
            p = A @ r.T + Win @ u
            q = self.leak_rate * jnp.tanh(p) + (1 - self.leak_rate) * r
        except TypeError as err:
            print('A.shape = {}, s.shape = {}, Win.shape = {}, u.shape = {}'
                  ''.format(A.shape, r.shape, Win.shape, u.shape))
            raise Exception('Likely dimension mismatch.') from err

        return q

    def readout(self,
                rt: ArrayLike,
                Wout: ArrayLike | None = None,
                utm1: ArrayLike | None = None
                ) -> ArrayLike:
        """use Wout to map reservoir state to output

        Args:
            rt: Reservoir state(s), either passed as single time snapshot with
                size (reservoir_dim,) or as 2D array with reservoir_dim as 
                last dim (time_dim, reservoir_dim).
                or as matrix, with reservoir dimension as last index
            Wout: Reservoir output weight matrix, size (reservoir_dim,
                input_dim). If None, will use self.Wout.
            utm1: 1D or 2D with size (u_dim,) or (time_dim, u_dim)
                u(t-1) for r(t). Only used if readout_method = 'biased',
                then Wout*[1, u(t-1), r(t)]=u(t)

        Returns:
            1D or 2D array with size (system_dim,) or (time_dim, system_dim)
            depending on shape of input array

        Todo:
            generalize similar to DiffRC
        """
        if Wout is None:
            Wout = self.Wout

        # necessary to copy in order to not
        # assign input reservoir state in place
        if self.readout_method == 'quadratic':
            st = deepcopy(rt)
            st = st.at[..., 1::2].set(st[..., 1::2]**2)
        elif self.readout_method == 'biased':
            assert utm1 is not None
            if rt.ndim > 1:
                st = jnp.concatenate((jnp.ones((rt.shape[0], 1)), utm1, rt),
                                     axis=1)
            else:
                st = jnp.concatenate(([1.0], utm1, rt))
        # linear
        else:
            st = rt

        try:
            vt = st @ Wout
        except TypeError:
            print('st.shape = {}, Wout.shape = {}'.format(
                st.shape, Wout.shape))
            raise

        return vt

    def train(self,
              state_vec: xr.Dataset,
              update_Wout: bool = True):
        """Train the localized RC model

        Args:
            state_vec: Training data with size (time_dim, output_dim)
            update_Wout: if True, update Wout, otherwise
                initialize it by rewriting the ybar and sbar matrices

        Sets Attributes:
            Wout (ArrayLike): Trained output weight matrix
        """

        r = self.readin(state_vec)['r'].data
        y = state_vec.to_array().stack(system=['variable','index']).data
        self.Wout = self._compute_Wout(r, y, update_Wout=update_Wout, u=y.T)

    def _compute_Wout(self,
                      rt: ArrayLike,
                      y: ArrayLike,
                      update_Wout: bool = True,
                      u: ArrayLike | None = None
                      ) -> ArrayLike:
        """Solve linear system with multiple RHS for readout weight matrix

        Args:
            rt: Reservoir states, size (time_dim, reservoir_dim),
            y: Target reservoir ouptut, size (time_dim, output_dim),
            update_Wout (bool): if True, update Wout, otherwise,
                initialize it by rewriting the ybar and sbar matrices

        Returns:
            Wout array, size (output_dim, reservoir_dim),
            If this is also stored within the object

        Sets Attributes:
            ybar (ArrayLike): y.T @ st, st is rt with readout_method accounted
                for.
            sbar (ArrayLike): st.T @ st, st is rt with readout_method
                accounted for.
            Wout (ArrayLike): see Returns.
            y_last, s_last, u_last (ArrayLike): the last element of output,
                reservoir, and input states
        """
        # Prepare for nonlinear readout function:
        # necessary to copy in order to not
        # assign input reservoir state in place
        if self.readout_method == 'quadratic':
            st = deepcopy(rt)
            st[...,  1::2] = st[..., 1::2]**2
        elif self.readout_method == 'biased':
            assert u is not None
            st = np.concatenate(
                    (np.ones((rt.shape[0]-1, 1)), u[:-1, :], rt[1:, :]),
                    axis=1)
            y = y[1:]
        else:
            st = rt

        # Learn weights by solving for final layer weights Wout analytically
        # (this is actually a linear regression, a no-hidden layer neural
        #  network with the identity activation function)
        if update_Wout:
            self.ybar = self.ybar + np.dot(y.T, st)
            self.sbar = self.sbar + np.dot(st.T, st)
        else:
            self.ybar = y @ st  # Changed from np.dot
            self.sbar = st.T @ st  # Changed from np.dot

        self.Wout = self._linsolve(
                (self.sbar
                 + self.tikhonov_parameter
                 * np.eye(self.sbar.shape[0])),
                self.ybar)
        
        # These are from the old update_Wout method,
        # although I'm not sure what they're for
        self.y_last = y[-1, ...]
        self.s_last = rt[-1, ...]
        if u is not None:
            self.u_last = u[-1, ...]

        return self.Wout

    def _linsolve(self,
                  X: ArrayLike,
                  Y: ArrayLike,
                  beta: float | None =  None,
                  **kwargs
                  ) -> ArrayLike:
        '''Linear solver wrapper for A in Y = AX

        Args:
            X: independent variable, square matrix
            Y: dependent variable, square matrix
            beta: Tikhonov regularization
        
        Returns: 
            Solution matrix, rectangular matrix
        '''
        A = self._linsolve_pinv(X, Y, beta)

        return A.T

    def _linsolve_pinv(self, 
                  X: ArrayLike,
                  Y: ArrayLike,
                  beta: float | None = None,
                  ) -> ArrayLike:
        """Solve for A in Y = AX, assuming X and Y are known.

        Args:
          X : independent variable, square matrix
          Y : dependent variable, square matrix
          beta: Tikhonov regularization

        Returns:
            Solution matrix, rectangular matrix
        """
        if beta is not None:
            Xinv = linalg.pinv(X+beta*np.eye(X.shape[0]))
        else:
            Xinv = linalg.pinv(X)
        A = Y @ Xinv

        return A

    def forecast(self,
                 res_state_vec: xr.Dataset,
                 n_steps: int =  1
                 ) -> xr.Dataset:
        """Run reservoir prediction from single reservoir state

        Notes:
            This is the method called by dab.dacycler's cycle() method.
            It performs the forecast in the reservoir space. self.readout()
            must be applied to the output to convert to system space.

        Args:
            res_state_vec: Xarray dataset containing reservoir state as
                data_var 'r'
            n_steps: Number of prediction steps
        
        Returns:
            Tuple of (last reservoir state, all reservoir states for n_steps).

        """
        if n_steps == 1:
            new_vals = self.update(res_state_vec['r'].data,
                                   self.readout(res_state_vec['r'].data))
            new_vec = xr.Dataset(
                {'r':(('time','reservoir'), new_vals)}
            )
        else:
            r = res_state_vec['r'].data
            r_full = jnp.zeros((n_steps, self.reservoir_dim))
            for i in range(n_steps):
                r_full = r_full.at[i].set(r)
                if i < n_steps:
                    r = self.update(r, self.readout(r))

            new_vec = xr.Dataset(
                {'r':(('time','reservoir'), r_full)}
            )
        return new_vec.isel(time=-1), new_vec

    def save_weights(self,
                     pkl_path: str):
        """Save RC reservoir weights as pkl file.

        Args:
            pkl_path: Filepath for saving with .pkl extension
        
        """
        with open(pkl_path, 'wb') as pkl:
            pickle.dump(self.Wout, pkl)

    def load_weights(self,
                     pkl_path: str):
        """Load RC reservoir weights from pkl file.

        Args:
            pkl_path: Filepath with save weight matrix.

        Sets Attributes:
            Wout (np.ndarray): Output weight matrix
        """
        with open(pkl_path, 'rb') as pkl:
            self.Wout = pickle.load(pkl)
        
