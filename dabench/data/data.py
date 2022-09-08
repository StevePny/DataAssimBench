"""Base class for data generator objects"""
import jax.numpy as jnp
from dabench.support.utils import integrate


class Data():
    """Generic class for data generator objects

    Attributes:
        system_dim (int): system dimension
        time_dim (int): total time steps
        original_dim (tuple): dimensions in original space, e.g. could be 3x3
            for a 2d system with system_dim = 9. Defaults to (system_dim),
            i.e. 1d.
        random_seed (int): random seed, defaults to 37
        delta_t (float): the timestep of the data (assumed uniform)
        values (:obj: `ndarray`): 2d array of data (time_dim, system_dim)
        """

    def __init__(self,
                 system_dim=3,
                 time_dim=1,
                 original_dim=None,
                 random_seed=37,
                 delta_t=0.01,
                 values=None,
                 **kwargs):
        """Initializes the base data object"""

        self.system_dim = system_dim
        self.time_dim = time_dim
        self.random_seed = random_seed
        self.values = values
        self.delta_t = delta_t

        if original_dim is None:
            self.original_dim = (system_dim)
        else:
            self.original_dim = original_dim

    def to_original_dim(self):
        """Converts 1D representation of system back to original dimensions

        Returns:
            Multidimensional array with shape:
            (time_dim, original_dim[0], ..., original_dim[n])
        """
        return jnp.reshape(self.values, (self.time_dim,) + self.original_dim)

    def sample_cells(self, targets):
        """Samples values at a list of multidimensional array indices.

        Args:
            targets (ndarray): Array of target indices in shape:
                (num_of_target_indices, time_dim + original_dim). E.g.
                [[0,0], [0,1]] samples the first and second cell values in the
                first timestep (in this case original_dim = 1).
        """
        tupled_targets = tuple(tuple(targets[:, i]) for
                               i in range(len(self.original_dim) + 1))
        return self.to_original_dim()[tupled_targets]

    def generate(self, n_steps=None, t_final=None, x0=None, M0=None,
                 return_tlm=False, stride=None, **kwargs):
        """Generates a dataset and assigns values and times to the data object.

        Notes:
            Either provide n_steps or t_final in order to indicate the length
            of the forecast. These are used to set the values, times, and
            time_dim attributes

        Args:
            n_steps (int): Number of timesteps. One of n_steps OR
                t_final must be specified.
            t_final (float): Final time of trajectory. One of n_steps OR
                t_final must be specified.
            x0 (ndarray): initial conditions state vector of shape (system_dim)
            M0 (ndarray): the initial condition of the TLM matrix computation
                shape (system_dim, system_dim)
            return_tlm (bool): specifies whether to compute and return the
                integrated Jacobian as a Tangent Linear Model for each timestep
            stride (int): specify how many steps to skip in the output data 
                versus the model timestep (delta_t)
            **kwargs: arguments to the integrate function (permits changes in
                convergence tolerance, etc.)

        Returns:
            Nothing if return_tlm=False. If return_tlm=True, returns tuple
                (xaux, M) where xaux is the system trajectory and M is a list
                of TLMs corresponding to the system trajectory.

        """

        # Checks
        if n_steps is not None:
            t_final = n_steps * self.delta_t
        elif t_final is not None:
            n_steps = int(t_final/self.delta_t)  # (not used here)
        else:
            raise Exception('Either n_steps or t_final must be supplied as an input argument.')

        if x0 is None:
            if self.x0 is not None:
                x0 = self.x0
            else:
                raise Exception('Initial condition is None,x0 = {}), it must either be provided as an argument or set as an attribute in the model object.'.format(x0))

        if return_tlm:
            if M0 is None:
                M0 = jnp.identity(self.system_dim)
            try:
                xaux0 = jnp.concatenate((x0.ravel(), M0.ravel()))
            except:  # Should add the specific error we're trying to catch here
                print('x0.shape = {}, M0.shape = {}'.format(x0.shape, M0.shape))
                raise Exception('EXITING...')
            x0 = xaux0
            if self.rhs_aux is None:
                raise Exception('self.rhs_aux must be specified prior to calling generate.')
            f = self.rhs_aux
        else:
            if self.rhs is None:
                raise Exception('self.rhs must be specified prior to calling generate.')
            f = self.rhs

        # Integrate and store values and times
        # If data object has it's own integration method, use that
        if hasattr(self, 'integrate') and callable(getattr(self, 'integrate')):
            y, t = self.integrate(f, x0, t_final, self.delta_t, stride=stride,
                                  **kwargs)
        # Otherwise, use integrate from dabench.support.utils
        else:
            y, t = integrate(f, x0, t_final, self.delta_t, stride=stride,
                             **kwargs)

        # The generate method specifically stores data in the object,
        # as opposed to the forecast method, which does not.
        # Store values and times as part of data object
        self.values = y[:self.system_dim, :]
        self.times = t
        self.time_dim = len(t)

        # Return the data series and associated TLMs if requested
        if return_tlm:
            # Reshape output
            xaux = y[:self.system_dim, :]

            # Initialize output matrices
            M = jnp.zeros((self.time_dim, self.system_dim, self.system_dim))

            # ISSUE: this can probably be formed by reshaping directly instead of a loop
            for i in range(self.time_dim):
                M[i, :, :] = jnp.reshape(y[self.system_dim:, i],
                                         (self.system_dim, self.system_dim))

            return xaux, M
