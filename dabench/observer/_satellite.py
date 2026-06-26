"""Satellite swath observation geometry for polar-orbiting instruments.

Models the moving ground-track footprint of a sun-synchronous (or
otherwise inclined) polar-orbiter constellation, so observation
networks can mimic real meteorological satellites instead of static
random masks.  Each satellite has a constant cross-track swath; over a
timestep its nadir track sweeps a wide stripe from near one pole
through the equator to near the other.  The mask at step ``t`` is the
union of all swath footprints traced over that step for every active
satellite, intersected with a polar-cap cut ``|phi| <= polar_cutoff_deg``.

The module is grid-agnostic: it operates on 1-D longitude/latitude
coordinate arrays (degrees) rather than any particular model grid, so
it can drive an :class:`~dabench.observer.Observer` over any gridded
:class:`~dabench.data.Data` whose nodal coordinates are known.

Instrument presets
------------------
``SATELLITE_PRESETS`` documents the modelled satellite for each swath
type (inclination, orbital period, cross-track swath half-width).  Pass
``instrument="viirs"`` (default), ``"modis"`` or ``"altimeter"``, or
override any orbital parameter explicitly.

Orbit kinematics (per satellite)
--------------------------------
Sun-synchronous, near-polar.  In an Earth-centered inertial frame with
the ascending node fixed at longitude ``initial_lon0_deg`` for sat 0::

  nu(tau)        = nu_0 + (360 deg / T_orb) * tau          (true anomaly)
  phi_sat(tau)   = arcsin( sin(i) * sin(nu(tau)) )         (geodetic lat)
  lambda_inert   = atan2( cos(i) * sin(nu), cos(nu) )      (in-plane lon)
  lambda_sat     = lambda_inert - omega_E * tau + lambda_0 (Earth-fixed)

For a constellation of ``N`` satellites the per-sat initial true
anomaly is ``nu_s0 = 360 deg * s / N`` (uniform spacing).  RAAN drift
under sun-sync (~1 deg/day) is neglected over short windows.
"""
from __future__ import annotations

from typing import List

import numpy as np

# For typing
ArrayLike = np.ndarray

_DEG2RAD = np.pi / 180.0
_EARTH_ROTATION_DEG_PER_H = 15.0   # sidereal rate (close enough at this res)

# Documented orbital parameters for the supported swath types.  Each entry
# names the real instrument it models so callers can pick a realistic
# geometry; any field can be overridden via satellite_swath_masks kwargs.
SATELLITE_PRESETS = {
    "viirs": {
        "inclination_deg": 98.7,
        "period_min": 101.5,
        "swath_halfwidth_deg": 13.7,
        "description": (
            "Suomi NPP / NOAA-20 / NOAA-21 VIIRS imager: ~3060 km "
            "cross-track at 824 km altitude, sun-synchronous."),
    },
    "modis": {
        "inclination_deg": 98.2,
        "period_min": 98.8,
        "swath_halfwidth_deg": 10.8,
        "description": (
            "Terra / Aqua MODIS imager: ~2330 km cross-track at 705 km "
            "altitude, sun-synchronous."),
    },
    "altimeter": {
        "inclination_deg": 66.0,
        "period_min": 112.4,
        "swath_halfwidth_deg": 0.06,
        "description": (
            "Jason-3 / Sentinel-6 Michael Freilich nadir radar altimeter: "
            "near-nadir ground track, 66 deg (non-sun-synchronous) orbit."),
    },
}


def _nadir_positions(
    *, t_steps: int, n_substeps_per_step: int, n_sats: int,
    inclination_deg: float, period_min: float, initial_lon0_deg: float,
    earth_rotation_deg_per_h: float, step_hours: float,
) -> np.ndarray:
    """Compute satellite nadir (lon, lat) at every substep, in degrees.

    Returns ``nadirs`` of shape ``(t_steps, n_substeps_per_step, n_sats, 2)``
    where the trailing axis is ``(lon_deg, lat_deg)`` in Earth-fixed
    coordinates, with longitude wrapped to ``[-180, 180]``.

    The substep at index ``s`` of step ``t`` corresponds to model time
    ``(t - 1 + (s + 1) / n_sub) * step_hours`` hours (right-closed,
    clamped to >= 0), so substep ``n_sub - 1`` lands exactly on step ``t``.
    """
    i_rad = inclination_deg * _DEG2RAD
    sin_i, cos_i = np.sin(i_rad), np.cos(i_rad)
    orbital_rate_deg_per_h = 60.0 * 360.0 / period_min
    nadirs = np.zeros((t_steps, n_substeps_per_step, n_sats, 2),
                      dtype=np.float64)
    nu_initial_deg = np.array([360.0 * s / n_sats for s in range(n_sats)])
    for t in range(t_steps):
        for s_idx in range(n_substeps_per_step):
            tau_h = (float(t - 1)
                     + float(s_idx + 1) / float(n_substeps_per_step)
                     ) * float(step_hours)
            tau_h = max(tau_h, 0.0)
            for k in range(n_sats):
                nu_deg = nu_initial_deg[k] + orbital_rate_deg_per_h * tau_h
                nu_rad = nu_deg * _DEG2RAD
                sin_nu, cos_nu = np.sin(nu_rad), np.cos(nu_rad)
                lat_rad = np.arcsin(sin_i * sin_nu)
                lon_inertial_rad = np.arctan2(cos_i * sin_nu, cos_nu)
                lon_earthfixed_deg = (
                    lon_inertial_rad / _DEG2RAD
                    - earth_rotation_deg_per_h * tau_h
                    + initial_lon0_deg)
                lon_wrapped = ((lon_earthfixed_deg + 180.0) % 360.0) - 180.0
                nadirs[t, s_idx, k, 0] = lon_wrapped
                nadirs[t, s_idx, k, 1] = lat_rad / _DEG2RAD
    return nadirs


def _great_circle_distance_deg(
    lon1_deg: ArrayLike, lat1_deg: ArrayLike,
    lon2_deg: ArrayLike, lat2_deg: ArrayLike,
) -> np.ndarray:
    """Great-circle distance in degrees via the haversine formula.

    Numerically stable down to sub-degree separations (avoids the
    ``arccos`` cancellation at small angles).  Inputs are broadcastable
    arrays of degrees; output matches the broadcast shape, in degrees.
    """
    lon1, lat1 = lon1_deg * _DEG2RAD, lat1_deg * _DEG2RAD
    lon2, lat2 = lon2_deg * _DEG2RAD, lat2_deg * _DEG2RAD
    dlat, dlon = lat2 - lat1, lon2 - lon1
    a = (np.sin(dlat / 2.0) ** 2
         + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2.0) ** 2)
    a = np.clip(a, 0.0, 1.0)
    return 2.0 * np.arcsin(np.sqrt(a)) / _DEG2RAD


def _thin_min_separation(
    mask2d: np.ndarray, lon_1d: np.ndarray, lat_1d: np.ndarray,
    thin_deg: float,
) -> np.ndarray:
    """Greedy min great-circle separation (superob) thinning of a swath mask.

    Visits the in-mask grid points in deterministic longitude-major order
    and keeps a point only when it lies at least ``thin_deg`` (great-circle
    degrees) from every already-kept point.  Returns a new boolean mask of
    the same shape; a no-op when ``thin_deg <= 0`` or the mask is empty.
    """
    if thin_deg <= 0.0:
        return mask2d
    ii, jj = np.nonzero(mask2d)
    if ii.size == 0:
        return mask2d
    out = np.zeros_like(mask2d, dtype=bool)
    keep_lon: List[float] = []
    keep_lat: List[float] = []
    for i, j in zip(ii.tolist(), jj.tolist()):
        lo, la = float(lon_1d[i]), float(lat_1d[j])
        if keep_lon:
            d = _great_circle_distance_deg(
                np.asarray(keep_lon), np.asarray(keep_lat), lo, la)
            if d.min() < thin_deg:
                continue
        keep_lon.append(lo)
        keep_lat.append(la)
        out[i, j] = True
    return out


def satellite_swath_masks(
    lon_deg: ArrayLike,
    lat_deg: ArrayLike,
    *,
    instrument: str = "viirs",
    n_sats: int = 1,
    t_steps: int = 12,
    step_hours: float = 1.0,
    n_substeps_per_step: int = 12,
    polar_cutoff_deg: float = 70.0,
    initial_lon0_deg: float = 0.0,
    inclination_deg: float | None = None,
    period_min: float | None = None,
    swath_halfwidth_deg: float | None = None,
    earth_rotation_deg_per_h: float = _EARTH_ROTATION_DEG_PER_H,
    thin_deg: float = 0.0,
) -> np.ndarray:
    """Build the swept-swath mask trajectory for a satellite constellation.

    Args:
        lon_deg: 1-D longitude coordinate array (degrees, any wrap).
        lat_deg: 1-D latitude coordinate array (degrees).
        instrument: Named swath type in :data:`SATELLITE_PRESETS`
            (``"viirs"`` default, ``"modis"``, ``"altimeter"``).  Sets the
            default inclination, period and swath half-width.
        n_sats: Number of satellites, uniformly spaced in initial true
            anomaly (``nu_s0 = 360 deg * s / n_sats``).
        t_steps: Number of timesteps in the assimilation window.
        step_hours: Duration of each timestep in hours.  Used together
            with the orbital rate so a 24-hour window can target full
            non-polar coverage regardless of the model's native cadence.
        n_substeps_per_step: Substeps used to discretise the swept ground
            track per timestep.
        polar_cutoff_deg: Latitude band cut; the mask is 0 outside
            ``|phi| <= polar_cutoff_deg`` (excludes over-sampled poles).
        initial_lon0_deg: Earth-fixed longitude of satellite 0's
            ascending node at ``t = 0``.
        inclination_deg, period_min, swath_halfwidth_deg: Explicit orbital
            overrides; default to the chosen ``instrument`` preset.
        earth_rotation_deg_per_h: Earth rotation rate (default sidereal).

    Returns:
        ``masks``: ``(t_steps, n_lon, n_lat)`` float32 binary masks
        (1.0 inside the swath, 0.0 outside).  The nodal axes are ordered
        longitude-first to match the standard ``(n_lon * n_lat)`` flatten.
    """
    if instrument not in SATELLITE_PRESETS:
        raise ValueError(
            f"unknown instrument {instrument!r}; choose from "
            f"{list(SATELLITE_PRESETS)} or pass explicit orbital params.")
    if n_sats < 1:
        raise ValueError(f"n_sats must be >= 1, got {n_sats}")
    if t_steps < 1:
        raise ValueError(f"t_steps must be >= 1, got {t_steps}")
    preset = SATELLITE_PRESETS[instrument]
    incl = preset["inclination_deg"] if inclination_deg is None \
        else float(inclination_deg)
    period = preset["period_min"] if period_min is None else float(period_min)
    halfwidth = preset["swath_halfwidth_deg"] if swath_halfwidth_deg is None \
        else float(swath_halfwidth_deg)

    lon_1d = ((np.asarray(lon_deg, dtype=np.float64) + 180.0) % 360.0) - 180.0
    lat_1d = np.asarray(lat_deg, dtype=np.float64)
    n_lon, n_lat = lon_1d.shape[0], lat_1d.shape[0]
    lon_grid = lon_1d[:, None]                  # (n_lon, 1)
    lat_grid = lat_1d[None, :]                  # (1, n_lat)

    nadirs = _nadir_positions(
        t_steps=t_steps, n_substeps_per_step=n_substeps_per_step,
        n_sats=n_sats, inclination_deg=incl, period_min=period,
        initial_lon0_deg=initial_lon0_deg,
        earth_rotation_deg_per_h=earth_rotation_deg_per_h,
        step_hours=step_hours)

    masks = np.zeros((t_steps, n_lon, n_lat), dtype=np.float32)
    polar_keep = np.abs(lat_grid) <= polar_cutoff_deg
    for t in range(t_steps):
        nadir_lon = nadirs[t, :, :, 0].ravel()
        nadir_lat = nadirs[t, :, :, 1].ravel()
        d = _great_circle_distance_deg(
            lon_grid[..., None], lat_grid[..., None],
            nadir_lon[None, None, :], nadir_lat[None, None, :])
        in_swath = d.min(axis=-1) <= halfwidth
        keep = in_swath & polar_keep
        if thin_deg > 0.0:
            keep = _thin_min_separation(keep, lon_1d, lat_1d, float(thin_deg))
        masks[t] = keep.astype(np.float32)
    return masks


def swath_location_sets(masks: ArrayLike) -> List[np.ndarray]:
    """Convert a mask trajectory to per-step flattened location indices.

    Args:
        masks: ``(t_steps, n_lon, n_lat)`` binary mask trajectory from
            :func:`satellite_swath_masks`.

    Returns:
        List of ``t_steps`` integer arrays; entry ``t`` holds the
        flattened ``(n_lon * n_lat)`` indices (longitude-major,
        ``idx = i_lon * n_lat + j_lat``) of the in-swath grid points at
        step ``t``.  Suitable as the per-time ``locations`` for a
        non-stationary :class:`~dabench.observer.Observer`.
    """
    masks_np = np.asarray(masks)
    t_steps = masks_np.shape[0]
    flat = masks_np.reshape(t_steps, -1)
    return [np.flatnonzero(flat[t] > 0).astype(np.int64)
            for t in range(t_steps)]


def coverage_summary(masks: ArrayLike) -> dict:
    """Quick coverage statistics for a mask trajectory.

    Args:
        masks: ``(t_steps, n_lon, n_lat)`` binary mask trajectory.

    Returns:
        dict with keys ``step_mean`` (per-step in-mask fraction,
        ``(t_steps,)``), ``step_mean_avg`` (float), ``union_fraction``
        (fraction hit by any step), and ``intersect_fraction``
        (fraction hit at every step).
    """
    masks_np = np.asarray(masks)
    t_steps, n_lon, n_lat = masks_np.shape
    n_total = float(n_lon * n_lat)
    per_step = masks_np.reshape(t_steps, -1).sum(axis=1) / n_total
    union = (masks_np.max(axis=0) > 0).sum() / n_total
    intersect = (masks_np.min(axis=0) > 0).sum() / n_total
    return {
        "step_mean": per_step,
        "step_mean_avg": float(per_step.mean()),
        "union_fraction": float(union),
        "intersect_fraction": float(intersect),
    }
