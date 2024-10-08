"""Tests for Observer class (dabench.observer._observer)"""

import pytest
import xarray as xr
import numpy as np
import jax.numpy as jnp

from dabench import observer, data, vector


def test_obs_l63():
    """Tests observer for Lorenz63"""
    l63 = data.Lorenz63()
    
    ds = l63.generate(n_steps=10)
    obs = observer.Observer(
            ds,
            random_time_density=0.5,
            random_location_density=0.5,
            error_sd=0.7,
            random_seed=99)

    obs_vec = obs.observe()

    assert obs_vec['x'].values.shape[0] == 8
    assert obs_vec['time'].shape[0] == 8
    assert np.array_equal(obs_vec['system_index'].values,
                          np.repeat(1, 8).reshape((1, 8, 1)))
    assert obs_vec['x'].values[0, 0] == pytest.approx(
            ds['x'].values[0, 1] + obs_vec.errors[0, 0, 0])
    assert np.allclose(obs_vec['x'].values, np.array([
        [-14.77003751], [-14.82969835], [-16.49873315], [-16.21225914],
        [-16.08467213], [-16.69890358], [-15.45879053], [-15.12257015]]
        )
    )
    assert np.allclose(obs_vec['errors'].values, np.array([[
        [ 0.22996249], [ 0.65467059], [-0.6143291 ], [-0.03212726],
        [ 0.26731022], [-0.31677728], [ 0.50516542], [-0.24651439]]]
        )
    )
    assert np.array_equal(obs_vec['time'].values, np.array(
        [0., 0.01, 0.02, 0.03, 0.04, 0.05, 0.07, 0.09]))



def test_obs_l63_count():
    """Tests observer for Lorenz63 using random_[time/location]_count"""
    l63 = data.Lorenz63()
    ds = l63.generate(n_steps=10)

    obs = observer.Observer(
            ds,
            random_time_count=5,
            random_location_count=2,
            error_sd=0.7)
    obs_vec = obs.observe()

    assert obs_vec['x'].shape[0] == 5
    assert obs_vec['time'].shape[0] == 5
    assert np.array_equal(obs_vec['system_index'],
                          np.tile([1, 2], 5).reshape((1, 5, 2)))
    assert obs_vec['x'].values[0, 0] == pytest.approx(
            ds['x'].values[1, 1] + obs_vec['errors'].values[0, 0, 0])
    assert np.allclose(obs_vec['x'].values, np.array([
        [-16.71412359,  23.46140116],
        [-16.5006219,  24.10746168],
        [-17.11500315,  27.69793923],
        [-15.78352562,  29.22759993],
        [-14.87744996,  31.11579368]]))
    assert np.allclose(obs_vec['errors'].values[0], np.array([
        [-1.22975335,  1.17910212],
        [-0.32048999, -0.41749407],
        [-0.73287729,  0.65225442],
        [0.47248634,  0.87110884],
        [0.6251612,  0.18410346]]))
    assert np.array_equal(obs_vec['time'], np.array(
        [0.01, 0.03, 0.05, 0.06, 0.08]))


def test_obs_l63_diffseed():
    """Tests observer for Lorenz63 with different seed and no errors"""

    l63 = data.Lorenz63()
    ds = l63.generate(n_steps=10)

    obs = observer.Observer(
            ds,
            random_time_density=0.5,
            random_location_density=0.5,
            error_sd=0.0,
            random_seed=1)
    obs_vec = obs.observe()

    assert obs_vec['x'].values.shape[0] == 5
    assert obs_vec['time'].shape[0] == 5
    assert np.array_equal(obs_vec['system_index'],
                          np.tile([0, 2], 5).reshape((1, 5, 2)))
    assert obs_vec['x'].values[0, 0] == ds['x'].values[0, 0]
    assert np.allclose(obs_vec['x'].values,
                       ds.sel(time=obs_vec['time'].values, index=[0,2]
                              )['x'].values)
    assert np.array_equal(obs_vec['time'],
                          np.array([0., 0.01, 0.03, 0.06, 0.08]))
    assert np.array_equal(obs_vec['errors'],
                          np.repeat(0, 10).reshape(1, 5, 2))


def test_obs_l63_specific_locs():
    """Tests observer for Lorenz63 with user-specified loations and times"""

    l63 = data.Lorenz63()
    ds = l63.generate(n_steps=10)

    obs = observer.Observer(
            ds,
            locations={'index':xr.DataArray([2],dims='observations')},
            times=ds['time'].values[[4, 9]],
            error_sd=1.0)
    obs_vec = obs.observe()

    assert obs_vec['x'].shape[0] == 2
    assert obs_vec['time'].shape[0] == 2
    assert np.array_equal(obs_vec['system_index'],
                          np.repeat(2, 2).reshape((1, 2, 1)))
    assert np.allclose(obs_vec['x'].values - obs_vec['errors'].values,
                       np.array([[ds['x'].values[4, 2]], [ds['x'].values[9, 2]]]))
    assert np.array_equal(obs_vec['time'],
                          np.array([0.04, 0.09]))
    assert np.allclose(obs_vec['errors'],
                       np.array([[0.0824943], [-0.46441841]]))


def test_obs_l96():
    """Tests observer for Lorenz96"""
    l96 = data.Lorenz96()
    ds = l96.generate(n_steps=10)

    obs = observer.Observer(
            ds,
            random_time_density=0.4,
            random_location_density=0.2,
            error_sd=0.7)
    obs_vec = obs.observe()

    assert obs_vec['x'].shape[0] == 3
    assert obs_vec['time'].shape[0] == 3
    assert np.array_equal(obs_vec['system_index'].values,
                          np.tile([16, 9, 31, 32], 3).reshape((1, 3, 4)))
    assert obs_vec['x'].values[0, 0] == pytest.approx(
            ds['x'].values[3, 16] + obs_vec['errors'].values[0, 0, 0])
    assert np.allclose(obs_vec['x'].values[0],
                       np.array([6.09641725, 2.80249778,
                                 2.8026745 , 6.91684992]))
    assert np.allclose(obs_vec['errors'].values[0,0],
                       np.array([ 0.5913821 , -0.22663247,
                                 0.00787655, -0.29022197]))
    assert np.allclose(obs_vec['time'], np.array([0.15, 0.2, 0.45]))


def test_obs_l96_moving():
    """Tests non-stationary observer for Lorenz96"""
    l96 = data.Lorenz96()
    ds = l96.generate(n_steps=10)

    obs = observer.Observer(
            ds,
            random_time_density=0.4,
            random_location_density=0.3,
            error_sd=1.2,
            stationary_observers=False)
    obs_vec = obs.observe()

    assert obs_vec['x'].shape == (3, 12)
    assert obs_vec['time'].shape[0] == 3
    assert obs_vec['system_index'].values[0, 2, -1] == 32
    assert obs_vec['x'].values[0, 0] == pytest.approx(
            ds['x'].values[3, 10] + obs_vec['errors'].values[0, 0, 0])
    assert obs_vec['x'].values[2, 5] == pytest.approx(0.010236507277464613)
    assert obs_vec['errors'].values[0, 1, 4] == pytest.approx(-0.7470301078422978)
    assert np.allclose(obs_vec['time'], np.array([0.15, 0.2, 0.45]))


def test_obs_pyqg():
    """Tests observer for PyQG, NOTE: skipping while pyqg is broken"""
    pytest.importorskip("pyqg")

    pyqg = data.PyQG()

    pyqg.generate(n_steps=10)

    obs = observer.Observer(
        pyqg,
        random_time_density=0.45,
        random_location_density=0.05,
        error_sd=1e-5)
    obs_vec = obs.observe()

    assert obs_vec.values.shape[0] == 6
    assert obs_vec.num_obs == 6
    assert obs_vec.times.shape[0] == 6
    assert obs_vec.time_indices.shape[0] == 6
    assert np.array_equal(obs_vec.obs_dims, np.repeat(431, 6))
    assert obs_vec.location_indices[0, 0] == 29
    assert obs_vec.values[0, 0] == pytest.approx(
            pyqg.values[1, 29] + obs_vec.errors[0, 0])
    assert obs_vec.values[2, 64] == pytest.approx(-6.591154124480716e-05)
    assert obs_vec.errors[2, 411] == pytest.approx(-6.324655243224687e-06)
    assert np.array_equal(obs_vec.times,
                          np.array([7200, 21600, 28800, 36000, 50400, 64800]))


def test_obs_gcp():
    """Tests observer for GCP downloaded ERA5 data"""
    gcp = data.GCP(date_start='2010-01-01',
                   date_end='2010-01-01')
    ds = gcp.load()

    obs = observer.Observer(
        ds,
        random_time_count=3,
        random_location_count=50,
        error_sd=0.0,
        error_bias=5.)
    obs_vec = obs.observe()
    obs_vec_flat = obs_vec.drop_dims('variable').dab.flatten()

    assert obs_vec_flat.shape == (3, 50)
    assert obs_vec['time'].shape[0] == 3
    assert obs_vec['system_index'].values[0, 0, 0] == 326
    assert obs_vec_flat.values[0, 0] == pytest.approx(
            ds.sel(time=obs_vec_flat['time'][0]
                   ).drop_vars('time').dab.flatten().values[326]
            + 5)
    assert obs_vec_flat[2, 42].values == pytest.approx(304.60122681)
    assert np.array_equal(obs_vec['errors'],
                          np.repeat(5, 3*50).reshape(1, 3, 50))
    assert obs_vec['time'][1] == np.datetime64('2010-01-01T18:00:00.000000000')


def test_obs_sqgturb():
    """Tests observer for SQGTurb"""
    sqg = data.SQGTurb()
    ds = sqg.generate(n_steps=10)

    obs = observer.Observer(
        ds,
        random_time_density=0.3,
        random_location_density=0.01,
        error_sd=25.)
    obs_vec = obs.observe()

    assert obs_vec['pv'].shape == (2, 204)
    assert obs_vec['time'].shape[0] == 2
    assert obs_vec['system_index'][0, 0, 0] == 10130
    assert obs_vec['pv'].values[1, 123] == pytest.approx(
            (ds.dab.flatten().sel(time=obs_vec['time'][1])[
                obs_vec['system_index'][0, 1, 123]]
             + obs_vec['errors'][0, 1, 123]))
    assert obs_vec['pv'].values[0, 44] == pytest.approx(2128.20749725)
    assert obs_vec['errors'].values[0, 1, 187] == pytest.approx(25.714826330507496)
    assert np.allclose(obs_vec['time'], np.array([2700., 9000.]))
