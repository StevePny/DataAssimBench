"""Hybrid in-situ + satellite observation network builder.

Assembles a single, reusable observation set that combines two
complementary observing systems over a gridded nature run:

* a *stationary* in-situ network whose stations are concentrated around
  a latitude band (e.g. a midlatitude jet), via
  :func:`~dabench.observer.jet_concentrated_indices`; and
* a *non-stationary* polar-orbiter swath whose footprint sweeps the
  globe over the assimilation window, via
  :func:`~dabench.observer.satellite_swath_masks`.

The two are merged per timestep (in-situ stations are always active;
the swath contributes the grid points inside the hour's footprint,
de-duplicated against the in-situ set) and handed to a non-stationary
:class:`~dabench.observer.Observer`, so the resulting object is a
standard DABench observation ``xr.Dataset`` that any cycler can consume.

Per-observation metadata is preserved so downstream code (and saved
files) know, for every observation: its **type** (``obs_type``: in-situ
vs satellite), its **source** instrument (dataset attrs), its **clean**
and **noisy** values (``<var>_clean`` data vars vs the noised data
vars), and its **prescribed error characteristics** (``obs_error_sd``
data var plus distribution/bias attrs).

Grid convention
---------------
The builder is grid-agnostic but assumes the flattened location
coordinate is **longitude-major**, ``idx = i_lon * n_lat + j_lat``,
matching :func:`jet_concentrated_indices` and
:func:`satellite_swath_masks`.  ``state_vec`` must carry a
``system_dim`` attr equal to ``n_lon * n_lat`` (set automatically if
absent).
"""
from __future__ import annotations

import numpy as np
import xarray as xr

from ._observer import Observer
from ._insitu import jet_concentrated_indices
from ._satellite import (
    SATELLITE_PRESETS,
    coverage_summary,
    satellite_swath_masks,
    swath_location_sets,
)

# obs_type codes stored in the returned dataset's ``obs_type`` data var.
OBS_TYPE_PADDED = 0
OBS_TYPE_INSITU = 1
OBS_TYPE_SATELLITE = 2


def build_hybrid_network(
    state_vec: xr.Dataset,
    lon_deg,
    lat_deg,
    *,
    n_insitu: int,
    jet_center_deg: float,
    jet_sigma_deg: float,
    uniform_floor: float = 0.2,
    instrument: str = "viirs",
    n_sats: int = 2,
    polar_cutoff_deg: float = 70.0,
    step_hours: float = 1.0,
    initial_lon0_deg: float = 0.0,
    insitu_error_sd: float = 0.0,
    satellite_error_sd: float = 0.0,
    error_bias: float = 0.0,
    error_positive_only: bool = False,
    times=None,
    location_coord: str = "index",
    time_coord: str = "time",
    random_seed: int = 99,
    store_as_jax: bool = False,
    fixed_pool: bool = False,
    swath_thin_deg: float = 0.0,
    swath_halfwidth_deg: float | None = None,
    error_sd_sys=None,
) -> xr.Dataset:
    """Build a combined jet-concentrated + moving-swath observation set.

    Args:
        state_vec: Gridded nature-run ``xr.Dataset`` to observe.  Must
            have a time coordinate (``time_coord``) and a flattened nodal
            location coordinate (``location_coord``) of length
            ``n_lon * n_lat``, longitude-major.
        lon_deg: 1-D longitude coordinate array (degrees), length n_lon.
        lat_deg: 1-D latitude coordinate array (degrees), length n_lat.
        n_insitu: Number of stationary in-situ stations to seed.
        jet_center_deg: Latitude of the in-situ Gaussian peak (degrees).
        jet_sigma_deg: Latitude std-dev of the in-situ Gaussian (degrees).
        uniform_floor: Uniform mixing fraction for the in-situ draw, in
            ``[0, 1]`` (see :func:`jet_concentrated_indices`).
        instrument: Swath preset name (see :data:`SATELLITE_PRESETS`).
        n_sats: Number of satellites in the constellation.
        polar_cutoff_deg: Latitude cut for the swath footprint.
        step_hours: Hours per timestep (drives orbital sweep rate so a
            24-step window can reach full non-polar coverage).
        initial_lon0_deg: Earth-fixed ascending-node longitude of sat 0.
        insitu_error_sd: Gaussian noise std for in-situ observations.
        satellite_error_sd: Gaussian noise std for satellite observations.
        error_bias: Mean of the Gaussian observation error.
        error_positive_only: Clip sampled errors to be non-negative.
        times: Observation times (defaults to all of ``state_vec``'s).
        location_coord: Name of the flattened nodal coordinate.
        time_coord: Name of the time coordinate.
        random_seed: Seed for the in-situ draw and the noise.
        store_as_jax: Store observation values as jax arrays.
        fixed_pool: Representation of the moving swath.  ``False``
            (default) emits the general non-stationary schema where the
            ``observations`` slot at each time holds a *different* grid
            location (per-time ``system_index``).  ``True`` emits a
            *fixed* observation pool (constant slot-to-location mapping:
            the union of the jet stations and every swath footprint),
            observed by a stationary :class:`Observer`, with the moving
            swath expressed purely through the per-step ``obs_active``
            mask (jet slots always active; a swath slot active only at
            steps when it is under the footprint) and inactive values
            NaN-masked.  Use ``True`` for cyclers that need a single
            fixed observation operator ``H`` plus a per-step mask.
        error_sd_sys: Optional per-system-index (length ``system_dim``)
            observation-error std array.  When given it OVERRIDES the
            scalar in-situ/satellite scatter, so the injected noise is
            per-grid-point; when ``None`` (default) the byte-identical
            scalar per-type behaviour is used.

    Returns:
        An observation ``xr.Dataset`` (DABench schema) with added
        per-observation metadata: ``obs_type``, ``obs_active``,
        ``obs_error_sd`` and ``<var>_clean`` data vars, plus
        network/instrument/error attrs and the swath
        ``coverage_union_fraction``.  When ``fixed_pool`` the dataset
        carries a constant observation pool with a ``pool_index`` coord
        (flattened grid index per slot) and ``stationary_observers`` is
        ``True``; otherwise the schema is non-stationary.
    """
    if instrument not in SATELLITE_PRESETS:
        raise ValueError(
            f"unknown instrument {instrument!r}; choose from "
            f"{list(SATELLITE_PRESETS)}.")

    lon_1d = np.asarray(lon_deg, dtype=np.float64)
    lat_1d = np.asarray(lat_deg, dtype=np.float64)
    n_lon, n_lat = lon_1d.shape[0], lat_1d.shape[0]
    grid_dim = int(n_lon * n_lat)
    system_dim = int(state_vec.sizes[location_coord])
    if grid_dim != system_dim:
        raise ValueError(
            f"grid_dim n_lon*n_lat={grid_dim} != state_vec "
            f"'{location_coord}' size {system_dim}.")
    if "system_dim" not in state_vec.attrs:
        state_vec = state_vec.assign_attrs(system_dim=grid_dim)

    if times is None:
        times = np.asarray(state_vec[time_coord].data)
    else:
        times = np.asarray(times)
    t_steps = int(times.shape[0])

    rng = np.random.default_rng(random_seed)
    worker = _assemble_fixed_pool if fixed_pool else _assemble
    return worker(
        state_vec, lon_1d, lat_1d, times, t_steps, grid_dim, system_dim,
        rng=rng, n_insitu=n_insitu, jet_center_deg=jet_center_deg,
        jet_sigma_deg=jet_sigma_deg, uniform_floor=uniform_floor,
        instrument=instrument, n_sats=n_sats,
        polar_cutoff_deg=polar_cutoff_deg, step_hours=step_hours,
        initial_lon0_deg=initial_lon0_deg, insitu_error_sd=insitu_error_sd,
        satellite_error_sd=satellite_error_sd, error_bias=error_bias,
        error_positive_only=error_positive_only,
        location_coord=location_coord, time_coord=time_coord,
        random_seed=random_seed, store_as_jax=store_as_jax,
        swath_thin_deg=swath_thin_deg, swath_halfwidth_deg=swath_halfwidth_deg,
        error_sd_sys=error_sd_sys)


def _assemble(
    state_vec, lon_1d, lat_1d, times, t_steps, grid_dim, system_dim, *,
    rng, n_insitu, jet_center_deg, jet_sigma_deg, uniform_floor,
    instrument, n_sats, polar_cutoff_deg, step_hours, initial_lon0_deg,
    insitu_error_sd, satellite_error_sd, error_bias, error_positive_only,
    location_coord, time_coord, random_seed, store_as_jax,
    swath_thin_deg=0.0, swath_halfwidth_deg=None, error_sd_sys=None,
) -> xr.Dataset:
    """Worker for :func:`build_hybrid_network` (kept short for clarity)."""
    # Stationary jet-concentrated in-situ stations.
    jet_idx = jet_concentrated_indices(
        lon_1d, lat_1d, int(n_insitu), jet_center_deg=jet_center_deg,
        jet_sigma_deg=jet_sigma_deg, uniform_floor=uniform_floor, rng=rng)

    # Per-step moving-swath footprints.
    masks = satellite_swath_masks(
        lon_1d, lat_1d, instrument=instrument, n_sats=int(n_sats),
        t_steps=t_steps, step_hours=step_hours,
        polar_cutoff_deg=polar_cutoff_deg, initial_lon0_deg=initial_lon0_deg,
        swath_halfwidth_deg=swath_halfwidth_deg, thin_deg=swath_thin_deg)
    swath_sets = swath_location_sets(masks)
    cov = coverage_summary(masks)

    # Merge per step: jet first, then swath minus jet (de-duplicated).
    coord_vals = np.asarray(state_vec[location_coord].data)
    locations, per_time_idx, per_time_type = [], [], []
    for t in range(t_steps):
        sw = np.asarray(swath_sets[t], dtype=np.int64)
        sw = sw[~np.isin(sw, jet_idx)]
        combined = np.concatenate([jet_idx, sw])
        types = np.concatenate([
            np.full(jet_idx.size, OBS_TYPE_INSITU, dtype=np.int8),
            np.full(sw.size, OBS_TYPE_SATELLITE, dtype=np.int8)])
        per_time_idx.append(combined)
        per_time_type.append(types)
        locations.append({location_coord: xr.DataArray(
            coord_vals[combined], dims=["observations"])})

    # Per-system-index error std: in-situ stations vs everything else.
    if error_sd_sys is not None:
        err_sd = np.asarray(error_sd_sys, dtype=np.float64)
        if err_sd.shape != (system_dim,):
            raise ValueError(f"error_sd_sys shape {err_sd.shape} != "
                             f"(system_dim={system_dim},)")
    else:                                    # byte-identical scalar per-type path
        err_sd = np.full(system_dim, float(satellite_error_sd), dtype=np.float64)
        err_sd[jet_idx] = float(insitu_error_sd)

    observer = Observer(
        state_vec, times=times, locations=locations,
        stationary_observers=False, error_bias=float(error_bias),
        error_sd=err_sd, error_positive_only=error_positive_only,
        random_seed=random_seed, store_as_jax=store_as_jax)
    obs_vec = observer.observe()

    return _attach_metadata(
        obs_vec, per_time_idx, per_time_type, err_sd, cov,
        t_steps=t_steps, time_coord=time_coord, instrument=instrument,
        n_sats=n_sats, polar_cutoff_deg=polar_cutoff_deg,
        jet_center_deg=jet_center_deg, jet_sigma_deg=jet_sigma_deg,
        uniform_floor=uniform_floor, n_insitu=int(n_insitu),
        error_bias=error_bias, insitu_error_sd=insitu_error_sd,
        satellite_error_sd=satellite_error_sd)


def _attach_metadata(
    obs_vec, per_time_idx, per_time_type, err_sd, cov, *,
    t_steps, time_coord, instrument, n_sats, polar_cutoff_deg,
    jet_center_deg, jet_sigma_deg, uniform_floor, n_insitu, error_bias,
    insitu_error_sd, satellite_error_sd,
) -> xr.Dataset:
    """Annotate the observed dataset with per-obs type/error metadata."""
    loc_dim = int(obs_vec.sizes["observations"])
    obs_type = np.zeros((t_steps, loc_dim), dtype=np.int8)
    obs_err = np.zeros((t_steps, loc_dim), dtype=np.float64)
    for t in range(t_steps):
        ty, idx = per_time_type[t], per_time_idx[t]
        obs_type[t, :ty.size] = ty
        obs_err[t, :idx.size] = err_sd[idx]

    # Recover clean (pre-noise) values per observed variable.
    data_vars = [str(v) for v in np.asarray(obs_vec["variable"].values)]
    for var in data_vars:
        clean = obs_vec[var] - obs_vec["errors"].sel(variable=var)
        obs_vec[f"{var}_clean"] = clean

    obs_vec = obs_vec.assign({
        "obs_type": ((time_coord, "observations"), obs_type),
        "obs_active": ((time_coord, "observations"), obs_type != 0),
        "obs_error_sd": ((time_coord, "observations"), obs_err),
    })
    obs_vec = obs_vec.assign_attrs(
        network_type="hybrid_insitu_satellite",
        instrument=instrument,
        instrument_description=SATELLITE_PRESETS[instrument]["description"],
        n_sats=int(n_sats),
        polar_cutoff_deg=float(polar_cutoff_deg),
        jet_center_deg=float(jet_center_deg),
        jet_sigma_deg=float(jet_sigma_deg),
        uniform_floor=float(uniform_floor),
        n_insitu=int(n_insitu),
        error_dist="gaussian",
        error_bias=float(error_bias),
        insitu_error_sd=float(insitu_error_sd),
        satellite_error_sd=float(satellite_error_sd),
        coverage_union_fraction=float(cov["union_fraction"]),
        obs_type_codes=(
            "0=padded/inactive, 1=insitu(jet), 2=satellite(swath)"),
    )
    return obs_vec


def _assemble_fixed_pool(
    state_vec, lon_1d, lat_1d, times, t_steps, grid_dim, system_dim, *,
    rng, n_insitu, jet_center_deg, jet_sigma_deg, uniform_floor,
    instrument, n_sats, polar_cutoff_deg, step_hours, initial_lon0_deg,
    insitu_error_sd, satellite_error_sd, error_bias, error_positive_only,
    location_coord, time_coord, random_seed, store_as_jax,
    swath_thin_deg=0.0, swath_halfwidth_deg=None, error_sd_sys=None,
) -> xr.Dataset:
    """Worker for ``build_hybrid_network(fixed_pool=True)``.

    Builds a constant observation pool (jet stations followed by the
    swath-only union, de-duplicated) observed by a *stationary*
    :class:`Observer`; the moving swath is carried entirely by the
    per-step ``obs_active`` mask, with inactive values NaN-masked.
    """
    jet_idx = jet_concentrated_indices(
        lon_1d, lat_1d, int(n_insitu), jet_center_deg=jet_center_deg,
        jet_sigma_deg=jet_sigma_deg, uniform_floor=uniform_floor, rng=rng)
    jet_idx = np.sort(jet_idx)

    masks = satellite_swath_masks(
        lon_1d, lat_1d, instrument=instrument, n_sats=int(n_sats),
        t_steps=t_steps, step_hours=step_hours,
        polar_cutoff_deg=polar_cutoff_deg, initial_lon0_deg=initial_lon0_deg,
        swath_halfwidth_deg=swath_halfwidth_deg, thin_deg=swath_thin_deg)
    swath_sets = swath_location_sets(masks)
    cov = coverage_summary(masks)

    nonempty = [s for s in swath_sets if s.size]
    union = (np.unique(np.concatenate(nonempty)) if nonempty
             else np.empty(0, dtype=np.int64))
    swath_only = np.setdiff1d(union, jet_idx, assume_unique=True)
    pool = np.concatenate([jet_idx, swath_only]).astype(np.int64)
    n_jet = int(jet_idx.size)

    # Per-system error std (in-situ on the jet sites, satellite elsewhere).
    if error_sd_sys is not None:
        err_sd_sys = np.asarray(error_sd_sys, dtype=np.float64)
        if err_sd_sys.shape != (system_dim,):
            raise ValueError(f"error_sd_sys shape {err_sd_sys.shape} != "
                             f"(system_dim={system_dim},)")
    else:                                    # byte-identical scalar per-type path
        err_sd_sys = np.full(system_dim, float(satellite_error_sd),
                             dtype=np.float64)
        err_sd_sys[jet_idx] = float(insitu_error_sd)

    coord_vals = np.asarray(state_vec[location_coord].data)
    locations = {location_coord: xr.DataArray(
        coord_vals[pool], dims=["observations"])}
    observer = Observer(
        state_vec, times=times, locations=locations,
        stationary_observers=True, error_bias=float(error_bias),
        error_sd=err_sd_sys, error_positive_only=error_positive_only,
        random_seed=random_seed, store_as_jax=store_as_jax)
    obs_vec = observer.observe()

    # Per-step active mask: jet always on; a swath slot on only when overhead.
    active = np.zeros((t_steps, pool.size), dtype=bool)
    active[:, :n_jet] = True
    slot_of_loc = {int(loc): n_jet + k for k, loc in enumerate(swath_only)}
    for t in range(t_steps):
        for loc in swath_sets[t]:
            slot = slot_of_loc.get(int(loc))
            if slot is not None:
                active[t, slot] = True

    pool_type = np.concatenate([
        np.full(n_jet, OBS_TYPE_INSITU, dtype=np.int8),
        np.full(swath_only.size, OBS_TYPE_SATELLITE, dtype=np.int8)])
    return _attach_metadata_fixed(
        obs_vec, pool, pool_type, err_sd_sys[pool], active, cov,
        time_coord=time_coord, instrument=instrument, n_sats=n_sats,
        polar_cutoff_deg=polar_cutoff_deg, jet_center_deg=jet_center_deg,
        jet_sigma_deg=jet_sigma_deg, uniform_floor=uniform_floor,
        n_insitu=int(n_insitu), error_bias=error_bias,
        insitu_error_sd=insitu_error_sd, satellite_error_sd=satellite_error_sd)


def _attach_metadata_fixed(
    obs_vec, pool, pool_type, pool_err, active, cov, *,
    time_coord, instrument, n_sats, polar_cutoff_deg, jet_center_deg,
    jet_sigma_deg, uniform_floor, n_insitu, error_bias, insitu_error_sd,
    satellite_error_sd,
) -> xr.Dataset:
    """Annotate a fixed-pool observed dataset and NaN-mask inactive obs."""
    obs_type = np.broadcast_to(pool_type[None, :], active.shape).copy()
    obs_err = np.where(active, pool_err[None, :], 0.0)

    # Clean (pre-noise) values per variable, recovered before masking.
    data_vars = [str(v) for v in np.asarray(obs_vec["variable"].values)]
    for var in data_vars:
        obs_vec[f"{var}_clean"] = obs_vec[var] - obs_vec["errors"].sel(
            variable=var)
    # NaN-mask inactive slots in both the noisy and clean values.
    for var in data_vars:
        for name in (var, f"{var}_clean"):
            da = obs_vec[name].transpose(time_coord, "observations")
            vals = np.array(da.data, dtype=np.float64)
            vals[~active] = np.nan
            obs_vec[name] = (da.dims, vals)

    obs_vec = obs_vec.assign_coords(
        pool_index=("observations", pool.astype(np.int64)))
    obs_vec = obs_vec.assign({
        "obs_type": ((time_coord, "observations"), obs_type),
        "obs_active": ((time_coord, "observations"), active),
        "obs_error_sd": ((time_coord, "observations"), obs_err),
    })
    # The moving swath is encoded purely as NaN-masked values, so the
    # dataset must advertise non-stationary observers (unless every slot is
    # active at every step) for cyclers to honour the per-step loc mask.
    obs_vec = obs_vec.assign_attrs(
        stationary_observers=bool(active.all()),
        network_type="hybrid_insitu_satellite",
        fixed_pool=True,
        instrument=instrument,
        instrument_description=SATELLITE_PRESETS[instrument]["description"],
        n_sats=int(n_sats),
        polar_cutoff_deg=float(polar_cutoff_deg),
        jet_center_deg=float(jet_center_deg),
        jet_sigma_deg=float(jet_sigma_deg),
        uniform_floor=float(uniform_floor),
        n_insitu=int(n_insitu),
        pool_size=int(pool.size),
        error_dist="gaussian",
        error_bias=float(error_bias),
        insitu_error_sd=float(insitu_error_sd),
        satellite_error_sd=float(satellite_error_sd),
        coverage_union_fraction=float(cov["union_fraction"]),
        obs_type_codes="1=insitu(jet), 2=satellite(swath); obs_active gates use",
    )
    return obs_vec
