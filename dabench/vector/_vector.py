"""Base class for vector objects"""

import copy

import numpy as np
import jax.numpy as jnp


class _Vector():
    """Base class for vector objects

    Attributes:
        system_dim (int): system dimension
        time_dim (int): total time steps
        delta_t (float): the timestep of the data (assumed uniform)
        values (ndarray): array of data values
        store_as_jax (bool): Store values as jax array instead of numpy array.
            Default is False (store as numpy).
        """

    def __init__(self,
                 time_dim=None,
                 times=None,
                 store_as_jax=False,
                 values=None):

        self.store_as_jax = store_as_jax
        self.values = values
        if times is None and self.values is not None:
            self.times = np.arange(self.values.shape[0])
        else:
            self.times = times
        if time_dim is None and self.times is not None:
            self.time_dim = self.times.shape[0]
        else:
            self.time_dim = time_dim

    def __getitem__(self, subscript):
        if self.values is None:
            raise AttributeError('Object does not contain any data values.\n'
                                 'Run .generate() or .load() and try again')

        if isinstance(subscript, slice):
            new_copy = copy.deepcopy(self)
            new_copy.values = new_copy.values[
                    subscript.start:subscript.stop:subscript.step]
            new_copy.times = new_copy.times[
                    subscript.start:subscript.stop:subscript.step]
            new_copy.time_dim = new_copy.times.shape[0]
            return new_copy
        else:
            new_copy = copy.deepcopy(self)
            new_copy.values = new_copy.values[subscript]
            new_copy.times = new_copy.times[subscript]
            if isinstance(subscript, int):
                new_copy.time_dim = 1
            else:
                new_copy.time_dim = new_copy.times.shape[0]
            return new_copy

    @property
    def values(self):
        return self._values

    @values.setter
    def values(self, vals):
        if vals is None:
            self._values = None
        else:
            if self.store_as_jax:
                self._values = jnp.asarray(vals)
            else:
                self._values = np.asarray(vals)

    @values.deleter
    def values(self):
        del self._values

    @property
    def times(self):
        return self._times

    @times.setter
    def times(self, vals):
        if vals is None:
            self._times = None
        else:
            if self.store_as_jax:
                self._times = jnp.asarray(vals)
            else:
                self._times = np.asarray(vals)

    @times.deleter
    def times(self):
        del self._times

    def filter_times(self, start, end, inclusive=True,
                     start_inclusive=None, end_inclusive=None):
        """Filter vector to within a time range, returns copy of object

        Args:
            start (datetime or float): Start of time range. Type must match
                type of times (e.g. datetime if times are datetimes, float if
                times are floats). Default is None, which includes all
                values up to "end". If neither start nor end are specified,
                time_filter does nothing.
            end (datetime or float): Start of time range. See "start" for
                more information. Default is None, which includes all values
                after "start".
            inclusive (bool): If True, includes times that are equal to start
                or end. If False, excludes times equal to start/end Default is
                True.
            start_inclusive, end_inclusive (bool):

        Returns:
            Copy of object with filtered values.
        """
        if start_inclusive is not None:
            inclusive = False
            if end_inclusive is None:
                end_inclusive = False
        if end_inclusive is not None:
            inclusive = False
            if start_inclusive is None:
                start_inclusive = False
        if inclusive:
            start_inclusive, end_inclusive = True, True

        if start is not None:
            if isinstance(start, np.datetime64):
                times_equal = (self.times == start)
            else:
                times_equal = np.isclose(self.times, start, rtol=0)
            if start_inclusive:
                filtered_idx = (self.times > start) + times_equal
            else:
                filtered_idx = (self.times > start) * ~times_equal
        else:
            filtered_idx = jnp.ones(self.times.shape[0])

        if end is not None:
            if isinstance(end, np.datetime64):
                times_equal = (self.times == end)
            else:
                times_equal = np.isclose(self.times, end, rtol=0)
            if end_inclusive:
                filtered_idx *= (self.times < end) + times_equal
            else:
                filtered_idx *= (self.times < end) * ~times_equal

        return self[filtered_idx]

