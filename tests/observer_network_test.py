"""Tests for the hybrid observation-network builder and its components
(dabench.observer._insitu, ._satellite, ._network)."""

import numpy as np
import xarray as xr

from dabench import observer


def _grid(n_lon=16, n_lat=12):
    lon = np.linspace(0.0, 360.0, n_lon, endpoint=False)
    lat = np.linspace(-82.5, 82.5, n_lat)
    return lon, lat


def _grid_dataset(n_lon=16, n_lat=12, t_steps=24, seed=0):
    lon, lat = _grid(n_lon, n_lat)
    grid_dim = n_lon * n_lat
    rng = np.random.default_rng(seed)
    x = rng.standard_normal((t_steps, grid_dim)).astype(np.float32)
    ds = xr.Dataset(
        {"x": (("time", "index"), x)},
        coords={"time": np.arange(t_steps, dtype=float),
                "index": np.arange(grid_dim, dtype=np.int64)},
        attrs={"system_dim": grid_dim, "delta_t": 1.0},
    )
    return ds, lon, lat


def test_jet_concentrated_indices_clusters_on_jet():
    """Pure-Gaussian draw should cluster near the jet latitude."""
    lon, lat = _grid(36, 37)
    n_lat = lat.shape[0]
    picked = observer.jet_concentrated_indices(
        lon, lat, 500, jet_center_deg=45.0, jet_sigma_deg=10.0,
        uniform_floor=0.0, rng=11)

    assert picked.shape == (500,)
    assert np.all((picked >= 0) & (picked < lon.shape[0] * n_lat))
    assert picked.size == np.unique(picked).size
    assert np.array_equal(picked, np.sort(picked))
    picked_lat = lat[picked % n_lat]
    # Mean latitude of the sample sits near the jet axis (~45 deg).
    assert abs(picked_lat.mean() - 45.0) < 8.0


def test_jet_uniform_floor_widens_spread():
    """A uniform floor pulls the sample mean back toward the equator."""
    lon, lat = _grid(36, 37)
    n_lat = lat.shape[0]
    pure = observer.jet_concentrated_indices(
        lon, lat, 500, jet_center_deg=45.0, jet_sigma_deg=8.0,
        uniform_floor=0.0, rng=3)
    mixed = observer.jet_concentrated_indices(
        lon, lat, 500, jet_center_deg=45.0, jet_sigma_deg=8.0,
        uniform_floor=0.9, rng=3)
    assert abs(lat[mixed % n_lat].mean()) < abs(lat[pure % n_lat].mean())


def test_satellite_swath_masks_shape_and_polar_cut():
    """Masks are binary, lon-major, and respect the polar cutoff."""
    lon, lat = _grid(36, 24)
    masks = observer.satellite_swath_masks(
        lon, lat, instrument="viirs", n_sats=2, t_steps=24,
        polar_cutoff_deg=70.0)
    assert masks.shape == (24, 36, 24)
    assert set(np.unique(masks)).issubset({0.0, 1.0})
    # No coverage outside the polar cap.
    polar = np.abs(lat) > 70.0
    assert masks[:, :, polar].sum() == 0.0


def test_swath_coverage_reaches_non_polar_by_24h():
    """Two VIIRS-like sats should blanket the non-polar band in 24 steps."""
    lon, lat = _grid(48, 36)
    masks = observer.satellite_swath_masks(
        lon, lat, instrument="viirs", n_sats=2, t_steps=24,
        step_hours=1.0, polar_cutoff_deg=70.0)
    cov = observer.coverage_summary(masks)
    non_polar_frac = float(np.mean(np.abs(lat) <= 70.0))
    # Union covers most of the reachable (non-polar) area.
    assert cov["union_fraction"] > 0.85 * non_polar_frac
    sets = observer.swath_location_sets(masks)
    assert len(sets) == 24
    flat0 = masks[0].reshape(-1)
    assert np.array_equal(sets[0], np.flatnonzero(flat0 > 0))


def test_build_hybrid_network_structure_and_metadata():
    """End-to-end builder: schema, obs_type, clean recovery, errors."""
    ds, lon, lat = _grid_dataset(n_lon=16, n_lat=12, t_steps=24)
    n_insitu = 20
    out = observer.build_hybrid_network(
        ds, lon, lat, n_insitu=n_insitu, jet_center_deg=45.0,
        jet_sigma_deg=15.0, instrument="viirs", n_sats=2,
        polar_cutoff_deg=70.0, step_hours=1.0, insitu_error_sd=0.1,
        satellite_error_sd=0.3, random_seed=7)

    assert out.sizes["time"] == 24
    ty = out["obs_type"].values
    assert set(np.unique(ty)).issubset(
        {observer.OBS_TYPE_PADDED, observer.OBS_TYPE_INSITU,
         observer.OBS_TYPE_SATELLITE})
    # In-situ stations are stationary: same count every step.
    assert np.all((ty == observer.OBS_TYPE_INSITU).sum(axis=1) == n_insitu)
    # Active flag matches the non-padded codes.
    assert np.array_equal(out["obs_active"].values, ty != 0)
    # Satellite footprint actually moves (varying active count per step).
    active_per_step = out["obs_active"].sum("observations").values
    assert active_per_step.max() > active_per_step.min()


def test_build_hybrid_network_clean_and_error_sd():
    """Noisy = clean + error; per-type error_sd is assigned correctly."""
    ds, lon, lat = _grid_dataset(n_lon=16, n_lat=12, t_steps=24)
    out = observer.build_hybrid_network(
        ds, lon, lat, n_insitu=20, jet_center_deg=45.0, jet_sigma_deg=15.0,
        instrument="viirs", n_sats=2, insitu_error_sd=0.1,
        satellite_error_sd=0.3, random_seed=7)

    ty = out["obs_type"].values
    act = out["obs_active"].values
    recon = out["x"].values - out["errors"].sel(variable="x").values
    assert np.allclose(recon[act], out["x_clean"].values[act], atol=1e-5)

    err_sd = out["obs_error_sd"].values
    assert np.allclose(err_sd[act & (ty == observer.OBS_TYPE_INSITU)], 0.1)
    assert np.allclose(err_sd[act & (ty == observer.OBS_TYPE_SATELLITE)], 0.3)
    assert out.attrs["network_type"] == "hybrid_insitu_satellite"
    assert out.attrs["instrument"] == "viirs"
    assert 0.0 < out.attrs["coverage_union_fraction"] <= 1.0


def test_build_hybrid_network_fixed_pool():
    """fixed_pool: constant slot->location, moving swath via obs_active mask."""
    ds, lon, lat = _grid_dataset(n_lon=16, n_lat=12, t_steps=24)
    out = observer.build_hybrid_network(
        ds, lon, lat, n_insitu=20, jet_center_deg=45.0, jet_sigma_deg=15.0,
        instrument="viirs", n_sats=2, polar_cutoff_deg=70.0, step_hours=1.0,
        insitu_error_sd=0.1, satellite_error_sd=0.3, random_seed=7,
        fixed_pool=True)

    assert out.attrs["fixed_pool"] is True
    # Moving swath -> must advertise non-stationary so cyclers honour NaNs.
    assert out.attrs["stationary_observers"] is False
    assert "pool_index" in out.coords
    assert out.coords["pool_index"].sizes["observations"] == out.attrs[
        "pool_size"]

    ty = out["obs_type"].values
    # Slot type is constant in time (fixed pool ordering).
    assert np.all(ty == ty[0][None, :])
    jet_slots = ty[0] == observer.OBS_TYPE_INSITU
    sat_slots = ty[0] == observer.OBS_TYPE_SATELLITE
    assert jet_slots.sum() == 20

    act = out["obs_active"].values
    # Jet slots active every step; satellite slots sweep (vary in time).
    assert np.all(act[:, jet_slots])
    if sat_slots.any():
        per_step = act[:, sat_slots].sum(axis=1)
        assert per_step.max() > per_step.min()

    # Inactive obs are NaN; active obs are finite and reconstruct the clean.
    x = out["x"].values
    assert np.all(np.isnan(x[~act]))
    assert np.all(np.isfinite(x[act]))
    recon = x - out["errors"].sel(variable="x").values
    assert np.allclose(recon[act], out["x_clean"].values[act], atol=1e-5)
    # pool_index slots are unique flattened grid indices.
    pool = out.coords["pool_index"].values
    assert pool.size == np.unique(pool).size


def test_network_seed_decouples_pool_from_noise_seed():
    """fixed_pool's total pool size is n_insitu + |swath minus in-situ
    overlap|, so it is a (weak) function of the in-situ random draw --
    i.e. of random_seed, when no separate network_seed is given.  Found
    via MLTLM's multi-anchor cotraining: each anchor uses a DIFFERENT
    random_seed (to decorrelate noise across anchors) while an ensemble
    cycler needs one FIXED observation dimension shared by every anchor.
    network_seed lets a caller vary random_seed (noise) while pinning the
    in-situ draw (and hence the pool) constant."""
    ds, lon, lat = _grid_dataset(n_lon=16, n_lat=12, t_steps=24)
    kw = dict(n_insitu=20, jet_center_deg=45.0, jet_sigma_deg=15.0,
             instrument="viirs", n_sats=2, polar_cutoff_deg=70.0,
             step_hours=1.0, insitu_error_sd=0.1, satellite_error_sd=0.3,
             fixed_pool=True)

    # Without network_seed: varying random_seed CAN change pool_size (the
    # regression this guards -- not guaranteed every seed pair differs,
    # but the whole point of network_seed is to make it never differ).
    out_a_pinned = observer.build_hybrid_network(
        ds, lon, lat, random_seed=1, network_seed=42, **kw)
    out_b_pinned = observer.build_hybrid_network(
        ds, lon, lat, random_seed=2, network_seed=42, **kw)
    assert out_a_pinned.attrs["pool_size"] == out_b_pinned.attrs["pool_size"]
    pool_a = out_a_pinned.coords["pool_index"].values
    pool_b = out_b_pinned.coords["pool_index"].values
    np.testing.assert_array_equal(pool_a, pool_b)  # same network, not just same size

    # But the actual observed (noised) values must still differ -- only
    # the network is pinned, not the noise realization.
    assert not np.allclose(
        np.nan_to_num(out_a_pinned["x"].values),
        np.nan_to_num(out_b_pinned["x"].values))

    # Default (network_seed=None) is BYTE-IDENTICAL to passing
    # network_seed=random_seed explicitly -- no behavior change for every
    # existing caller that never passes network_seed.
    out_default = observer.build_hybrid_network(
        ds, lon, lat, random_seed=5, **kw)
    out_explicit = observer.build_hybrid_network(
        ds, lon, lat, random_seed=5, network_seed=5, **kw)
    np.testing.assert_array_equal(
        out_default.coords["pool_index"].values,
        out_explicit.coords["pool_index"].values)
    np.testing.assert_array_equal(
        np.nan_to_num(out_default["x"].values),
        np.nan_to_num(out_explicit["x"].values))
