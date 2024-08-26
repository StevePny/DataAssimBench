"""Utils for data assimilation cyclers"""

import jax.numpy as jnp
import numpy as np


def _get_all_times(
    start_time,
    analysis_window,
    analysis_cycles,
    ):
    """Calculate times of the centers of all analysis windows.

    Args:
        start_time (float): Start time of DA experiment in model time units.
        analysis_window (float): Length of analysis window, in model time 
            units.
        analysis_cycles (int): Number of analysis cycles to perform.

    Returns:
        array of all analysis window center-times.

    
    """
    all_times = (
            jnp.repeat(start_time, analysis_cycles)
            + jnp.arange(0, analysis_cycles*analysis_window,
                            analysis_window)
                    )

    return all_times


def _get_obs_indices(
    analysis_times,
    obs_times,
    analysis_window,
    start_inclusive=True,
    end_inclusive=False
    ):
    """Get indices of obs times for each analysis cycle to pass to jax.lax.scan

    Args:
        analysis_times (list): List of times for all analysis window, centered
            in middle of time window. Output of _get_all_times().
        obs_times (list): List of times for all observations.
        analysis_window (float): Length of analysis window.
        start_inclusive (bool): Include obs times equal to beginning of 
            analysis window. Default is True
        end_inclusive (bool): Include obs times equal to end of 
            analysis window. Default is False.
    
    Returns:
        list with each element containing array of obs indices for the
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


def _pad_time_indices(
    obs_indices,
    add_one=True
    ):
    """Pad observation indices for each analysis window.

    Args:
        obs_indices (list): List of arrays where each array contains
            obs indices for an analysis cycle. Result of _get_obs_indices.
        add_one (bool): If True, will add one to all index values to encode
            indices to be masked out for DA (i.e. zeros represent indices to
            be masked out). Default is True.

    Returns:
        padded_indices (array): Array of padded obs_indices, with shape: 
            (num_analysis_cycles, max_obs_per_cycle).
    """

    def resize(row, size, add_one):
        new = np.array(row) + add_one
        new.resize(size)
        return new

    # find longest row length
    row_length = max(obs_indices, key=len).__len__()
    padded_indices = np.array([resize(row, row_length, add_one) for row in obs_indices])

    return padded_indices


def _pad_obs_locs(obs_vec):
    """Pad observation location indices to equal spacing

    Args:
        obs_vec (dabench.vector.ObsVector): Observation vector
            object containing times, locations, and values of obs.

    Returns:
        (vals, locs, masks): Tuple containing padded arrays of obs
            values and locations, and binary array masks where 1 is
            a valid observation value/location and 0 is not.
    """

    def resize(row, size):
        new_vals_locs = np.array(np.stack(row), order='F')
        new_vals_locs.resize((new_vals_locs.shape[0], size))
        mask = np.ones_like(new_vals_locs[0]).astype(int)
        if size > len(row[0]):
            mask[-(size-len(row[0])):] = 0
        return np.vstack([new_vals_locs, mask]).T

    # Find longest row length
    row_length = max(obs_vec.values, key=len).__len__()
    padded_arrays_masks = np.array([resize(row, row_length) for row in 
                                    np.stack([obs_vec.values,
                                              obs_vec.location_indices],
                                              axis=1)], dtype=float)
    vals, locs, masks = (padded_arrays_masks[...,0],
                         padded_arrays_masks[...,1:-1].astype(int),
                         padded_arrays_masks[...,2].astype(bool))
    if locs.shape[-1] == 1:
        locs = locs[..., 0]

    return vals, locs, masks