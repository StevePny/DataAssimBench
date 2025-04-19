"""Class for reservoir computing model for NWP"""

import logging
from copy import deepcopy
import pickle

from scipy import sparse, stats, linalg
import numpy as np
import jax.numpy as jnp
import xarray as xr

from dabench import vector, model

logging.basicConfig(filename='logfile.log', level=logging.DEBUG)


class RCModel(model.Model):
    """Class for a simple Reservoir Computing data-driven model

    Attributes:
        system_dim (int): Dimension of reservoir output.
        input_dim (int): Dimension of reservoir input signal.
        reservoir_dim (int): Dimension of reservoir state. Default: 512.

        sparsity (float): the percentage of zero-valued entries in the
            adjacency matrix (A). Default: 0.99.
        sparse_adj_matrix (bool): If True, A is computed using scipy sparse.
        sigma (float): Scaling of the input weight matrix. Default: 0.5.
        sigma_bias (float): Bias term for sigma. Default: 0.
        leak_rate (float): ``(1-leak_rate)`` of the reservoir state at the
            previous time step is incorporated during timestep update.
            Default: 1.
        spectral_radius (float): Scaling of the reservoir adjacency
            matrix. Default: 0.9.
        tikhonov_parameter (float): Regularization parameter in the linear
            solving, penalizing amplitude of weight matrix elements. Default: 0.0.
        readout_method (str): How to handle reservoir state elements during
            readout. One of 'linear', 'biased', or 'quadratic'.
            Default: 'linear'.

        random_seed (int): Random seed for random number generation. Default
            is 1.

        s (ndarray): Model states over entire time series.
        s_last (ndarray): Last
        ybar (ndarray): y.T @ st, set in _compute_Wout.
        sbar (ndarray): st.T @ st, set in _compute_Wout.
        A (ndarray): reservoir adjacency weight matrix, set in
            ``.weights_init()``.
        Win (ndarray): reservoir input weight matrix, set in
            ``.weights_init()``.
        Wout (ndarray): trained output weight matrix, set in ``.train()``.
    """

    def __init__(self,
                 system_dim,
                 input_dim,
                 reservoir_dim=512,
                 sparsity=0.99,
                 sparse_adj_matrix=False,
                 sigma=0.5,
                 sigma_bias=0,
                 leak_rate=1.0,
                 spectral_radius=0.9,
                 tikhonov_parameter=0,
                 readout_method='linear',
                 random_seed=1,
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
            A (array_like): (reservoir_dim, reservoir_dim),
                reservoir adjacency matrix
            Win (array_like): (reservoir_dim, input_dim),
                reservoir input weight matrix
            Adense (array_like): stores dense version of A if A is specified
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

    def generate(self, state_vec, A=None, Win=None, r0=None):
        """generate reservoir time series from input signal u

        Args:
            u (array_like): (time_dimension, system_dimension), input signal to
                reservoir
            A (array_like, optional): (reservoir_dim, reservoir_dim),
                reservoir adjacency matrix
            Win (array_like, optional): (reservoir_dim, system_dimension),
                reservoir input weight matrix
            r0 (array_like, optional): (reservoir_dim,) initial reservoir state
            save_states (bool): If True, saves reservoir states as self.states.
                If False, returns states. Default: False.

        Returns:
            r (array_like): (time_dim, reservoir_dim), reservoir state
        """
        u = state_vec.to_stacked_array('system',['time']).data
        r = np.zeros((u.shape[0], self.reservoir_dim))

        if r0 is not None:
            logging.debug(
                    'generate:: using initial reservoir state: %s', r0)
            r[0, :] = np.reshape(r0, (1, self.reservoir_dim))

        # Encoding input signal {u(t)} -> {s(t)}
        for t in range(0, u.shape[0]):
            r[t, :] = self.update(r[t - 1], u[t - 1, :], A, Win)

        return xr.Dataset(
            {'r': (('time', 'reservoir'), r)},
            coords={'time':state_vec.time}
        )

    def update(self, r, u, A=None, Win=None):
        """Update reservoir state with input signal and previous state

        Args:
            r (array_like): (reservoir_dim,) Previous reservoir state
            u (array_like): (input_dimension,) input signal
            A (array_like, optional): (reservoir_dim, reservoir_dim),
                reservoir adjacency matrix. If None, uses self.A. Default
                is None.
            Win (array_like, optional): (reservoir_dim, input_dimension),
                reservoir input weight matrix. If None, uses self.Win.
                Default is None
        Returns:
            q (array_like): (reservoir_dim,) Reservoir state at next time step
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

    def predict(self, state_vec, delta_t, initial_index=0, n_steps=100,
                spinup_steps=0, r0=None, keep_spinup=True):
        """Compute the prediction phase of the RC

        Args:
            dataobj (Data): data object containing the initial conditions
            initial_index (int, optional): time index of initial conditions in
                the data object 'values'
            n_steps (int, optional): number of steps to conduct the prediction
            spinup_steps (int, optional): number of steps before the
                initial_index to use for spinning up the reservoir state
            r0 (array_like, optional): initial reservoir state

        Returns:
            dataobj_pred (vector.StateVector): StateVector object covering
                prediction period
        """

        # Recompute the initial reservoir spinup to get reservoir states
        if spinup_steps > 0:
            u = state_vec.values[(initial_index-spinup_steps):initial_index]
            r = self.generate(u, r0=r0, save_states=False)
            r0 = r[-1, ]

        if r0 is not None:
            s_last = r0
        else:
            s_last = self.states[initial_index-1]

        u_last = state_vec.values[max(initial_index-1, 0), :]

        # Use these if possible
        A = getattr(self, 'A', None)
        Win = getattr(self, 'Win', None)
        predicted_obj = self._predict_backend(n_steps, s_last.T, u_last.T,
                                              delta_t, A=A, Win=Win)

        if keep_spinup and spinup_steps > 0:
            predicted_values = jnp.concatenate([u, predicted_obj.values])
            predicted_times = state_vec.times
        else:
            predicted_values = predicted_obj.values
            predicted_times = state_vec.times[initial_index:]

        out_vec = vector.StateVector(
                values=predicted_values,
                times=predicted_times,
                store_as_jax=True)

        return out_vec

    def readout(self, rt, Wout=None, utm1=None):
        """use Wout to map reservoir state to output

        Args:
            rt (array_like): 1D or 2D with dims: (Nr,) or (Ntime, Nr)
                reservoir state, either passed as single time snapshot,
                or as matrix, with reservoir dimension as last index
            utm1 (array_like): 1D or 2D with dims: (Nu,) or (Ntime, Nu)
                u(t-1) for r(t), only used if readout_method = 'biased',
                then Wout*[1, u(t-1), r(t)]=u(t)

        Returns:
            vt (array_like): 1D or 2D with dims: (Nout,) or (Ntime, Nout)
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

    def _predict_backend(self, n_samples, s_last, u_last, delta_t,
                         A=None, Win=None, Wout=None):
        """Apply the learned weights to new input.

        Args:
            n_samples (int): number of time steps to predict
            s_last (array_like): 1D vector with final reservoir state before
                prediction.
            u_last (array_like): 1D vector with final input signal before
                prediction.
            delta_t (float): full time length of spinup and prediction windows
            A (array_like, optional): (reservoir_dim, reservoir_dim),
                adjacency matrix. If None, uses self.A. Default is None.
            Win (array_like, optional): (reservoir_dim, input_dimension),
                input weight matrix. If None, uses self.Win.
                Default is None.
            Wout (array_like, optional): Rutput weight matrix. If None,
                uses self.Wout. Default is None.

        Returns:
            y (Data): data object with predicted signal from reservoir
        """

        s = jnp.zeros((n_samples, self.reservoir_dim))
        y = jnp.zeros((n_samples, self.system_dim))
        s = s.at[0].set(self.update(s_last, u_last, A, Win))
        y = y.at[0].set(self.readout(s[0, :], Wout, utm1=u_last))

        for t in range(n_samples - 1):
            s = s.at[t + 1].set(self.update(s[t, :], y[t, :], A, Win))
            y = y.at[t + 1].set(self.readout(s[t + 1, :], Wout, utm1=y[t, :]))

        y_obj = vector.StateVector(
            system_dim=y.shape[1],
            time_dim=y.shape[0],
            values=y,
            times=jnp.arange(1, y.shape[0]+1)*delta_t,
            store_as_jax=True)

        return y_obj

    def train(self, data_obj, update_Wout=True):
        """Train the localized RC model

        Args:
            dataobj (Data): Data object containing training data
            update_Wout (bool): if True, update Wout, otherwise
                initialize it by rewriting the ybar and sbar matrices

        Sets Attributes:
            Wout (array_like): Trained output weight matrix
        """

        r = self.generate(data_obj)['r'].data
        # u = data_obj.to_array().transpose(..., 'variable').data.reshape(data_obj.sizes['time'], -1)
        u = data_obj.to_array().stack(system=['variable','i']).data
        self.Wout = self._compute_Wout(r, u, update_Wout=update_Wout, u=u.T)

    def _compute_Wout(self, rt, y, update_Wout=True, u=None):
        """Solve linear system with multiple RHS for readout weight matrix

        Args:
            rt (array_like): 2D with dims (time_dim, reservoir_dim),
                reservoir state
            y (array_like): 2D with dims (time_dim, output_dim),
                target reservoir output
            update_Wout (bool): if True, update Wout, otherwise,
                initialize it by rewriting the ybar and sbar matrices

        Returns:
            Wout (array_like): 2D with dims (output_dim, reservoir_dim),
                this is also stored within the object

        Sets Attributes:
            ybar (array_like): y.T @ st, st is rt with readout_method accounted
                for.
            sbar (array_like): st.T @ st, st is rt with readout_method
                accounted for.
            Wout (array_like): see Returns.
            y_last, s_last, u_last (array_like): the last element of output,
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

    def _linsolve(self, X, Y, beta=None, **kwargs):
        '''Linear solver wrapper for A in Y = AX

        Args:
            X (matrix) : independent variable
            Y (matrix) : dependent variable
            beta (float): Tikhonov regularization
        '''
        A = self._linsolve_pinv(X, Y, beta)

        return A.T

    def _linsolve_pinv(self, X, Y, beta=None):
        """Solve for A in Y = AX, assuming X and Y are known.

        Args:
          X : independent variable, square matrix
          Y : dependent variable, square matrix

        Returns:
          A : Solution matrix, rectangular matrix
        """
        if beta is not None:
            Xinv = linalg.pinv(X+beta*np.eye(X.shape[0]))
        else:
            Xinv = linalg.pinv(X)
        A = Y @ Xinv

        return A

    def forecast(self, state_vec, n_steps=1):
        if n_steps == 1:
            new_vals = self.update(state_vec['r'].data,
                                   self.readout(state_vec['r'].data))
            new_vec = xr.Dataset(
                {'r':(('time','reservoir'), new_vals)}
            )
        else:
            r = state_vec['r'].data
            r_full = jnp.zeros((n_steps, self.reservoir_dim))
            for i in range(n_steps):
                r_full = r_full.at[i].set(r)
                if i < n_steps-1:
                    r = self.update(r, self.readout(r))

            new_vec = xr.Dataset(
                {'r':(('time','reservoir'), r_full)}
            )
        return new_vec.isel(time=-1), new_vec.drop_isel(time=-1)

    def save_weights(self, pkl_path):
        """Save RC reservoir weights as pkl file.

        Args:
            pkl_path (str): Filepath for saving with .pkl extension
        """
        with open(pkl_path, 'wb') as pkl:
            pickle.dump(self.Wout, pkl)

    def load_weights(self, pkl_path):
        """Load RC reservoir weights from pkl file.

        Args:
            pkl_path (str): Filepath with save weight matrix.
        """
        with open(pkl_path, 'rb') as pkl:
            self.Wout = pickle.load(pkl)
        
