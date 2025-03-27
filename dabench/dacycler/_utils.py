"""Utils for data assimilation cyclers"""

import jax.numpy as jnp
import jax
import numpy as np
import xarray as xr
import xarray_jax as xj


# For typing
ArrayLike = list | np.ndarray | jax.Array
XarrayDatasetLike = xr.Dataset | xj.XjDataset

def _get_all_times(
        start_time: float,
        analysis_window: float,
        analysis_cycles: int
        ) -> jax.Array:
    """Calculate times of the centers of all analysis windows.

    Args:
        start_time: Start time of DA experiment in model time units.
        analysis_window: Length of analysis window, in model time 
            units.
        analysis_cycles: Number of analysis cycles to perform.

    Returns:
        Array of all analysis window center-times.

    
    """
    all_times = (
            jnp.repeat(start_time, analysis_cycles)
            + jnp.arange(0, analysis_cycles*analysis_window,
                            analysis_window)
                    )

    return all_times


def _get_obs_indices(
        analysis_times: ArrayLike,
        obs_times: ArrayLike,
        analysis_window: float,
        start_inclusive: bool = True,
        end_inclusive: bool = False
        ) -> list:
    """Get indices of obs times for each analysis cycle to pass to jax.lax.scan

    Args:
        analysis_times: List of times for all analysis window, centered
            in middle of time window. Output of _get_all_times().
        obs_times: List of times for all observations.
        analysis_window: Length of analysis window.
        start_inclusive: Include obs times equal to beginning of 
            analysis window. Default is True
        end_inclusive: Include obs times equal to end of 
            analysis window. Default is False.
    
    Returns:
        List with each element containing array of obs indices for the
            corresponding analysis cycle.
    """
    # Get the obs vectors for each analysis window
    all_filtered_idx = [jnp.where(
        # Greater than start of window
        (obs_times > cur_time - analysis_window/2)
        # AND Less than end of window
        * (obs_times < cur_time + analysis_window/2)
        # AND not equal to start of window
        * (1-(1-start_inclusive)*jnp.isclose(obs_times, cur_time - analysis_window/2,
                                             rtol=0))
        # AND not equal to end of window
        * (1-(1-end_inclusive)*jnp.isclose(obs_times, cur_time + analysis_window/2,
                                           rtol=0))
        # OR Equal to start of window end
        + start_inclusive*jnp.isclose(obs_times, cur_time - analysis_window/2,
                                      rtol=0)
        # OR Equal to end of window
        + end_inclusive*jnp.isclose(obs_times, cur_time + analysis_window/2,
                                    rtol=0)
        )[0] for cur_time in analysis_times]

    return all_filtered_idx


def _time_resize(
        row: ArrayLike,
        size: int,
        add_one: bool
        ) -> np.ndarray:
    new = np.array(row) + add_one
    new.resize(size)
    return new


def _pad_time_indices(
        obs_indices: ArrayLike,
        add_one: bool = True
        ) -> ArrayLike:
    """Pad observation indices for each analysis window.

    Args:
        obs_indices: List of arrays where each array contains
            obs indices for an analysis cycle. Result of _get_obs_indices.
        add_one: If True, will add one to all index values to encode
            indices to be masked out for DA (i.e. zeros represent indices to
            be masked out). Default is True.

    Returns:
        padded_indices: Array of padded obs_indices, with shape: 
            (num_analysis_cycles, max_obs_per_cycle).
    """
    # find longest row length
    row_length = max(obs_indices, key=len).__len__()
    padded_indices = np.array([_time_resize(row, row_length, add_one)
                               for row in obs_indices])

    return padded_indices


def _obs_resize(
        row: ArrayLike,
        size: float
        ) -> np.ndarray:
    new_vals_locs = np.array(np.stack(row), order='F')
    new_vals_locs.resize((new_vals_locs.shape[0], size))
    mask = np.ones_like(new_vals_locs[0]).astype(int)
    if size > len(row[0]):
        mask[-(size-len(row[0])):] = 0
    return np.vstack([new_vals_locs, mask]).T


def _pad_obs_locs(
        obs_vec: XarrayDatasetLike
        ) -> tuple[ArrayLike, ArrayLike, ArrayLike]:
    """Pad observation location indices to equal spacing

    Args:
        obs_vec: Xarray containing times, locations, and values of obs.

    Returns:
        Tuple containing padded arrays of obs
            values and locations, and binary array masks where 1 is
            a valid observation value/location and 0 is not.
    """
    # Find longest row length
    row_length = max(obs_vec.values, key=len).__len__()
    padded_arrays_masks = np.array([_obs_resize(row, row_length) for row in
                                    np.stack([obs_vec.values,
                                              obs_vec.location_indices],
                                              axis=1)], dtype=float)
    vals, locs, masks = (padded_arrays_masks[...,0],
                         padded_arrays_masks[...,1:-1].astype(int),
                         padded_arrays_masks[...,2].astype(bool))
    if locs.shape[-1] == 1:
        locs = locs[..., 0]

    return vals, locs, masks