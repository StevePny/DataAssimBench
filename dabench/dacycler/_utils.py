"""Utils for data assimilation cyclers"""

import jax.numpy as jnp


def _get_all_times(
    start_time,
    analysis_window,
    analysis_cycles,
    analysis_time_in_window=None
    ):
    """Calculate times of the centers of all analysis windows.

    Args:
        start_time (float): Start time of DA experiment in model time units.
        analysis_window (float): Length of analysis window, in model time 
            units.
        analysis_cycles (int): Number of analysis cycles to perform.
        analysis_time_in_window (float): The time within the window on which
            each analysis cycle is centered. If None, uses the time halfway
            through the analysis window (i.e. analysis_window/2). Default is
            None.

    Returns:
        array of all analysis window center-times.

    
    """
    if analysis_time_in_window is None:
        analysis_time_in_window = analysis_window/2

    all_times = (
            jnp.repeat(start_time + analysis_time_in_window, analysis_cycles)
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
        # OR Equal to start of window end
        + start_inclusive*jnp.isclose(obs_times, cur_time - analysis_window/2,
                                      rtol=0)
        # OR Equal to end of window
        + end_inclusive*jnp.isclose(obs_times, cur_time + analysis_window/2,
                                    rtol=0)
        )[0] for cur_time in analysis_times]

    return all_filtered_idx


def _pad_indices(
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
        new = jnp.array(row) + add_one
        new.resize(size)
        return new

    # find longest row length
    row_length = max(obs_indices, key=len).__len__()
    padded_indices = jnp.array([resize(row, row_length, add_one) for row in obs_indices])

    return padded_indices
