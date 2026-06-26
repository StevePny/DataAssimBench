"""In-situ observation location sampling for gridded data.

Provides latitude-weighted station seeding for in-situ (point)
observation networks.  The motivating use case is a midlatitude jet:
real conventional networks (radiosondes, aircraft, surface stations)
are denser where the dynamically active flow lives, so concentrating
observation sites around the jet axis is more realistic — and more
informative for data assimilation — than a globally uniform draw.

The sampler weights each grid latitude by a Gaussian centred on a
specified latitude (e.g. the cache's jet axis) with a specified
latitude standard deviation, mixed with a uniform floor so the rest of
the globe is not left completely unobserved.  Longitudes are weighted
uniformly (a zonal jet has no preferred longitude).

The module is grid-agnostic: it works on 1-D longitude/latitude
coordinate arrays (degrees) and returns flattened ``(n_lon * n_lat)``
location indices (longitude-major, ``idx = i_lon * n_lat + j_lat``),
ready to drive an :class:`~dabench.observer.Observer`.
"""
from __future__ import annotations

import numpy as np

# For typing
ArrayLike = np.ndarray


def jet_concentrated_indices(
    lon_deg: ArrayLike,
    lat_deg: ArrayLike,
    n_obs: int,
    *,
    jet_center_deg: float,
    jet_sigma_deg: float,
    uniform_floor: float = 0.2,
    rng: np.random.Generator | int | None = None,
) -> np.ndarray:
    """Sample station indices concentrated around a latitude band.

    Each grid cell ``(i_lon, j_lat)`` is drawn without replacement with
    probability proportional to a latitude weight::

        w(phi) = (1 - uniform_floor) * exp(-0.5 * ((phi - mu) / sigma)^2)
                 + uniform_floor

    normalised over all cells (the weight is constant in longitude).  The
    ``uniform_floor`` term guarantees a non-zero probability everywhere so
    off-jet regions are not completely blind.

    Args:
        lon_deg: 1-D longitude coordinate array (degrees).
        lat_deg: 1-D latitude coordinate array (degrees).
        n_obs: Number of distinct station indices to draw.
        jet_center_deg: Latitude ``mu`` of the Gaussian peak (degrees);
            e.g. the cache's ``jet_lat0_deg``.
        jet_sigma_deg: Latitude standard deviation ``sigma`` (degrees);
            e.g. ``rad2deg(jet_width_rad)``.
        uniform_floor: Fraction of the (pre-normalisation) weight that is
            spatially uniform, in ``[0, 1]``.  ``0`` = pure Gaussian,
            ``1`` = globally uniform.  Default 0.2.
        rng: ``numpy.random.Generator``, integer seed, or ``None``.

    Returns:
        Sorted 1-D int64 array of ``n_obs`` flattened grid indices
        (longitude-major).

    Raises:
        ValueError: if ``n_obs`` exceeds the number of grid cells,
            ``jet_sigma_deg <= 0`` or ``uniform_floor`` is out of range.
    """
    if not 0.0 <= uniform_floor <= 1.0:
        raise ValueError(
            f"uniform_floor must be in [0, 1], got {uniform_floor}")
    if jet_sigma_deg <= 0.0:
        raise ValueError(f"jet_sigma_deg must be > 0, got {jet_sigma_deg}")

    lat_1d = np.asarray(lat_deg, dtype=np.float64)
    n_lon = int(np.asarray(lon_deg).shape[0])
    n_lat = int(lat_1d.shape[0])
    grid_dim = n_lon * n_lat
    if not 1 <= int(n_obs) <= grid_dim:
        raise ValueError(
            f"n_obs={n_obs} out of range for grid_dim={grid_dim}.")

    if not isinstance(rng, np.random.Generator):
        rng = np.random.default_rng(rng)

    # Per-latitude Gaussian weight + uniform floor (constant in longitude).
    z = (lat_1d - float(jet_center_deg)) / float(jet_sigma_deg)
    lat_w = (1.0 - uniform_floor) * np.exp(-0.5 * z * z) + uniform_floor
    # Broadcast to (n_lon, n_lat), flatten longitude-major, normalise.
    weights = np.broadcast_to(lat_w[None, :], (n_lon, n_lat)).reshape(grid_dim)
    weights = weights / weights.sum()

    picked = rng.choice(grid_dim, size=int(n_obs), replace=False, p=weights)
    return np.sort(picked).astype(np.int64)
