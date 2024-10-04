"""Ensemble forecast metrics"""

import jax.numpy as jnp
from dabench.metrics import _utils


__all__ = [
    'rank_histogram',
    'crps_ensemble',
    ]


def rank_histogram(observations, forecasts, dim=None, member_dim="member"):
    """JAX array implementation of Rank Histogram

    Description:
        (from https://www.cawcr.gov.au/projects/verification/#Methods_for_EPS)

        Answers the question: How well does the ensemble spread of the forecast represent the true variability (uncertainty) of the observations?

        Also known as a "Talagrand diagram", this method checks where the verifying observation usually falls with respect to the ensemble forecast data, which is arranged in increasing order at each grid point. In an ensemble with perfect spread, each member represents an equally likely scenario, so the observation is equally likely to fall between any two members.

        To construct a rank histogram, do the following:
        1. At every observation (or analysis) point rank the N ensemble members from lowest to highest. This represents N+1 possible bins that the observation could fit into, including the two extremes
        2. Identify which bin the observation falls into at each point
        3. Tally over many observations to create a histogram of rank.

        Interpretation:
        Flat - ensemble spread about right to represent forecast uncertainty
        U-shaped - ensemble spread too small, many observations falling outside the extremes of the ensemble
        Dome-shaped - ensemble spread too large, most observations falling near the center of the ensemble
        Asymmetric - ensemble contains bias

        Note: A flat rank histogram does not necessarily indicate a good forecast, it only measures whether the observed probability distribution is well represented by the ensemble.

    Args:
        predictions (ndarray): Array of predictions
        targets (ndarray): Array of targets to compare against. Shape must
            be broadcastable to shape of predictions.

    Returns:
        [UPDATE] Float, Pearson's R correlation coefficient.
    """

    # RMSD = sqrt( 1/(N+1) * sum(Sk - M/(N+1)^2) )

    # See: https://github.com/xarray-contrib/xskillscore/blob/64f17fdd1816b64b9e13c3f2febb9800a7e6ed0c/xskillscore/core/probabilistic.py#L830C20-L830C76

    def _rank_first(x, y):
        """Concatenates x and y and returns the rank of the
        first element along the last axes"""
        xy = jnp.concatenate((x[..., jnp.newaxis], y), axis=-1)
        return bn.nanrankdata(xy, axis=-1)[..., 0]

    if dim is not None:
        if len(dim) == 0:
            raise ValueError(
                "At least one dimension must be supplied to compute rank histogram over"
            )
        if member_dim in dim:
            raise ValueError(f'"{member_dim}" cannot be specified as an input to dim')

    ranks = xr.apply_ufunc(
        _rank_first,
        observations,
        forecasts,
        input_core_dims=[[], [member_dim]],
        dask="parallelized",
        output_dtypes=[int],
    )

    bin_edges = jnp.arange(0.5, len(forecasts[member_dim]) + 2)
    return histogram(ranks, bins=[bin_edges], bin_names=["rank"], dim=dim, bin_dim_suffix="")


def crps_ensemble(observations, forecasts, axis=-1):
    """JAX array implementation of Continuous Ranked Probability Score

    (From: https://confluence.ecmwf.int/display/FUG/Section+12.B+Statistical+Concepts+-+Probabilistic+Data#:~:text=The%20Continuous%20Ranked%20Probability%20Score,the%20forecast%20is%20wholly%20inaccurate.)

    A generalisation of Ranked Probability Score (RPS) is the Continuous Rank Probability Score (CRPSS) where the thresholds are continuous  rather than discrete (see Nurmi, 2003; Jollife and Stephenson, 2003; Wilks, 2006). The Continuous Ranked Probability Score (CRPS) is a measure of how good forecasts are in matching observed outcomes.   Where:

    CRPS = 0 the forecast is wholly accurate;
    CRPS = 1 the forecast is wholly inaccurate.
    CRPS is calculated by comparing the Cumulative Distribution Functions (CDF) for the forecast against a reference dataset (observations, or analyses, or climatology) over a given period.

    Args:
        predictions (ndarray): Array of predictions
        targets (ndarray): Array of targets to compare against. Shape must
            be broadcastable to shape of predictions.

    Returns:
        [UPDATE] Float, Mean Squared Error
    """

    # Integral from -inf to inf: (1/M) * sum[ S [P_j(x) - H(x - x_oj)]^2 dx ]
    # where Pj, H, and x_oj are the predicted cumulative distribution for case j, the Heaviside step function,
    # and the observed value, respectively.
    # (see: https://www.ecmwf.int/sites/default/files/elibrary/2007/10729-ensemble-forecasting.pdf)
    # with M independent cases (e.g. different dates)

    # See: https://github.com/properscoring/properscoring/blob/a465b5578d4b661e662933e84fa7673a70e75e94/properscoring/_crps.py#L244

    # Manage input quality
    observations = jnp.asarray(observations)
    forecasts = jnp.asarray(forecasts)

    if axis != -1:
        # Move the axis to the end
        forecasts = jnp.rollaxis(forecasts, axis, start=forecasts.ndim)

    if observations.shape not in [forecasts.shape, forecasts.shape[:-1]]:
        raise ValueError('observations and forecasts must have matching '
                         'shapes or matching shapes except along `axis=%s`'
                         % axis)

    if observations.shape == forecasts.shape:
        if weights is not None:
            raise ValueError('cannot supply weights unless you also supply '
                             'an ensemble forecast')
        return abs(observations - forecasts)

    # Sort forecast members by target quantity
    idx = jnp.argsort(forecasts, axis=-1)
    forecasts = forecasts[idx]
    weights = jnp.ones_like(forecasts)

    return _crps_ensemble_vectorized(observation, forecasts, weights, result)

#   @guvectorize(["void(float64[:], float64[:], float64[:], float64[:])"],
#            "(),(n),(n)->()", nopython=True)

    @partial(jnp.vectorize, signature='(),(n),(n)->()')
    def _crps_ensemble_vectorized(observation, forecasts, weights, result):
        # beware: forecasts are assumed sorted in NumPy's sort order

        # add asserts here:

        # we index the 0th element to get the scalar value from this 0d array:
        # http://numba.pydata.org/numba-doc/0.18.2/user/vectorize.html#the-guvectorize-decorator
        obs = observation[0]

        if jnp.isnan(obs):
            result[0] = jnp.nan
            return

        total_weight = 0.0
        for n, weight in enumerate(weights):
            if jnp.isnan(forecasts[n]):
                # NumPy sorts NaN to the end
                break
            if not weight >= 0:
                # this catches NaN weights
                result[0] = jnp.nan
                return
            total_weight += weight

        obs_cdf = 0
        forecast_cdf = 0
        prev_forecast = 0
        integral = 0

        for n, forecast in enumerate(forecasts):
            if jnp.isnan(forecast):
                # NumPy sorts NaN to the end
                if n == 0:
                    integral = jnp.nan
                # reset for the sake of the conditional below
                forecast = prev_forecast
                break

            if obs_cdf == 0 and obs < forecast:
                integral += (obs - prev_forecast) * forecast_cdf ** 2
                integral += (forecast - obs) * (forecast_cdf - 1) ** 2
                obs_cdf = 1
            else:
                integral += ((forecast - prev_forecast)
                             * (forecast_cdf - obs_cdf) ** 2)

            forecast_cdf += weights[n] / total_weight
            prev_forecast = forecast

        if obs_cdf == 0:
            # forecast can be undefined here if the loop body is never executed
            # (because forecasts have size 0), but don't worry about that because
            # we want to raise an error in that case, anyways
            integral += obs - forecast

        result[0] = integral


