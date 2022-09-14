"""Base class for data generator objects"""
import jax.numpy as jnp


class Data():
    """Generic class for data generator objects

    Attributes:
        system_dim (int): system dimension
        time_dim (int): total time steps
        original_dim (tuple): dimensions in original space, e.g. could be 3x3
            for a 2d system with system_dim = 9. Defaults to (system_dim),
            i.e. 1d.
        random_seed (int): random seed, defaults to 37
        values (:obj: `ndarray`): 2d array of data (time_dim, system_dim)
        """

    def __init__(self,
                 system_dim=3,
                 time_dim=1,
                 original_dim=None,
                 random_seed=37,
                 values=None,
                 **kwargs):
        """Initializes the base data object"""

        self.system_dim = system_dim
        self.time_dim = time_dim
        self.random_seed = random_seed
        self.values = values

        if original_dim is None:
            self.original_dim = (system_dim)
        else:
            self.original_dim = original_dim

    def set_values(self, values):
        """Sets values manually

        Args:
            values (ndarray): New values with shape (time_dim, system_dim).
        """
        self.values = values
        self.time_dim = values.shape[0]
        self.system_dim = values.shape[1]

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
            targets: Array of target indices in shape: (num_of_target_indices,
                time_dim + original_dim). E.g. [[0,0], [0,1]] samples the
                first and second cell values in the first timestep (in this
                case original_dim = 1).
        """
        tupled_targets = tuple(tuple(targets[:, i]) for
                               i in range(len(self.original_dim) + 1))
        return self.to_original_dim()[tupled_targets]
