import xarray as xr
import numpy as np
import warnings


def _check_split_lengths(xr_obj, split_lengths):
    total_length = np.sum(split_lengths)
    xr_timedim = xr_obj.sizes['time']
    if xr_timedim < total_length:
        warnings.warn("Specified split lengths ({}) exceed \n"
                      "Xarray object's time dimension ({}).".format(
                          split_lengths, xr_timedim
                      ))
    elif xr_timedim > total_length:
        warnings.warn("Specified split lengths ({}) are shorter than "
                      "Xarray object's time dimension ({}).".format(
                          split_lengths, xr_timedim
                      ))


@xr.register_dataset_accessor('dab')
class DABenchDatasetAccessor:
    """Helper methods for manipulating xarray Datasets"""
    def __init__(self, xarray_obj):
        self._obj = xarray_obj

    def flatten(self):
        if 'time' in self._obj.coords:
            remaining_dim = ['time']
        else:
            remaining_dim = []
        return self._obj.to_stacked_array('system', remaining_dim)

    def split_train_val_test(self, split_lengths):
        if (np.array(split_lengths) > 1.0).sum() == 0:
            # Assuming split_lengths is provided as fraction
            split_lengths = np.round(
                    np.array(split_lengths)*self._obj.sizes['time']
                    ).astype(int)
        _check_split_lengths(self._obj, split_lengths)
        out_ds = []
        start_i = 0
        for sl in split_lengths:
            end_i = start_i + sl
            out_ds.append(self._obj.isel(time=slice(start_i, end_i)))
        return tuple(out_ds)


@xr.register_dataarray_accessor('dab')
class DABenchDataArrayAccessor:
    """Helper methods for manipulating xarray DataArrays"""
    def __init__(self, xarray_obj):
        self._obj = xarray_obj

    def unflatten(self):
        return self._obj.to_unstacked_dataset('system')

    def split_train_val_test(self, split_lengths):
        if (np.array(split_lengths) > 1.0).sum() == 0:
            # Assuming split_lengths is provided as fraction
            split_lengths = np.round(
                    np.array(split_lengths)*self._obj.sizes['time']
                    ).astype(int)
        _check_split_lengths(self._obj, split_lengths)
        out_ds = []
        start_i = 0
        for sl in split_lengths:
            end_i = start_i + sl
            out_ds.append(self._obj.isel(time=slice(start_i, end_i)))
        return tuple(out_ds)


