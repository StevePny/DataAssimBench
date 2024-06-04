"""Tests for Observer class (dabench.observer._observer)"""

import pytest
import numpy as np
import jax.numpy as jnp

from dabench import observer, data, vector


def test_obs_l63():
    """Tests observer for Lorenz63"""
    l63 = data.Lorenz63()
    l63.generate(n_steps=10)

    obs = observer.Observer(
            l63,
            random_time_density=0.5,
            random_location_density=0.5,
            error_sd=0.7)
    obs_vec = obs.observe()

    assert obs_vec.values.shape[0] == 8
    assert obs_vec.num_obs == 8
    assert obs_vec.times.shape[0] == 8
    assert obs_vec.time_indices.shape[0] == 8
    assert np.array_equal(obs_vec.obs_dims, np.repeat(1, 8))
    assert np.array_equal(obs_vec.location_indices,
                          np.repeat(0, 8).reshape((8, 1)))
    assert obs_vec.values[0, 0] == pytest.approx(
            l63.values[0, 0] + obs_vec.errors[0, 0])
    assert np.allclose(obs_vec.values, np.array([
        [-9.81589654], [-10.26987252], [-10.33960451], [-12.08917307],
        [-11.96410835], [-12.08793537], [-13.37283807], [-12.98607508]]))
    assert np.allclose(obs_vec.errors, np.array([
        [0.18410346], [0.22996249], [0.65467059], [-0.6143291],
        [-0.03212726], [0.26731022], [-0.31677728], [0.50516542]]))
    assert np.array_equal(obs_vec.times, np.array(
        [0., 0.01, 0.02, 0.03, 0.04, 0.05, 0.07, 0.09]))


def test_obs_l63_count():
    """Tests observer for Lorenz63 using random_[time/location]_count"""
    l63 = data.Lorenz63()
    l63.generate(n_steps=10)

    obs = observer.Observer(
            l63,
            random_time_count=5,
            random_location_count=2,
            error_sd=0.7)
    obs_vec = obs.observe()

    assert obs_vec.values.shape[0] == 5
    assert obs_vec.num_obs == 5
    assert obs_vec.times.shape[0] == 5
    assert obs_vec.time_indices.shape[0] == 5
    assert np.array_equal(obs_vec.obs_dims, np.repeat(2, 5))
    assert np.array_equal(obs_vec.location_indices,
                          np.tile([1, 2], 5).reshape((5, 2)))
    assert obs_vec.values[0, 0] == pytest.approx(
            l63.values[1, 1] + obs_vec.errors[0, 0])
    assert np.allclose(obs_vec.values, np.array([
        [-16.71412359,  23.46140116],
        [-16.5006219,  24.10746168],
        [-17.11500315,  27.69793923],
        [-15.78352562,  29.22759993],
        [-14.87744996,  31.11579368]]))
    assert np.allclose(obs_vec.errors, np.array([
        [-1.22975335,  1.17910212],
        [-0.32048999, -0.41749407],
        [-0.73287729,  0.65225442],
        [0.47248634,  0.87110884],
        [0.6251612,  0.18410346]]))
    assert np.array_equal(obs_vec.times, np.array(
        [0.01, 0.03, 0.05, 0.06, 0.08]))


def test_obs_l63_diffseed():
    """Tests observer for Lorenz63 with different seed and no errors"""

    l63 = data.Lorenz63()
    l63.generate(n_steps=10)

    obs = observer.Observer(
            l63,
            random_time_density=0.5,
            random_location_density=0.5,
            error_sd=0.0,
            random_seed=1)
    obs_vec = obs.observe()

    assert obs_vec.values.shape[0] == 5
    assert obs_vec.num_obs == 5
    assert obs_vec.times.shape[0] == 5
    assert obs_vec.time_indices.shape[0] == 5
    assert np.array_equal(obs_vec.obs_dims, np.repeat(2, 5))
    assert np.array_equal(obs_vec.location_indices,
                          np.tile([0, 1], 5).reshape((5, 2)))
    assert obs_vec.values[0, 0] == l63.values[0, 0]
    assert np.allclose(obs_vec.values,
                       np.array([[-10., -15.],
                                 [-10.49983501, -15.48437023],
                                 [-11.47484398, -16.18013191],
                                 [-12.73363876, -16.25601196],
                                 [-13.31179237, -15.50261116]])
                       )
    assert np.array_equal(obs_vec.times,
                          np.array([0., 0.01, 0.03, 0.06, 0.08]))
    assert np.array_equal(obs_vec.errors,
                          np.repeat(0, 10).reshape(5, 2))


def test_obs_l63_specific_locs():
    """Tests observer for Lorenz63 with user-specified loations and times"""

    l63 = data.Lorenz63()
    l63.generate(n_steps=10)

    obs = observer.Observer(
            l63,
            location_indices=[2],
            time_indices=[4, 9],
            error_sd=1.0)
    obs_vec = obs.observe()

    assert obs_vec.values.shape[0] == 2
    assert obs_vec.num_obs == 2
    assert obs_vec.times.shape[0] == 2
    assert obs_vec.time_indices.shape[0] == 2
    assert np.array_equal(obs_vec.obs_dims, np.repeat(1, 2))
    assert np.array_equal(obs_vec.location_indices,
                          np.repeat(2, 2).reshape((2, 1)))
    assert np.allclose(obs_vec.values - obs_vec.errors,
                       np.array([[l63.values[4, 2]], [l63.values[9, 2]]]))
    assert np.array_equal(obs_vec.times,
                          np.array([0.04, 0.09]))
    assert np.allclose(obs_vec.errors,
                       np.array([[0.0824943], [-0.46441841]]))


def test_obs_l96():
    """Tests observer for Lorenz96"""
    l96 = data.Lorenz96()
    l96.generate(n_steps=10)

    obs = observer.Observer(
            l96,
            random_time_density=0.4,
            random_location_density=0.2,
            error_sd=0.7)
    obs_vec = obs.observe()

    assert obs_vec.values.shape[0] == 3
    assert obs_vec.num_obs == 3
    assert obs_vec.times.shape[0] == 3
    assert obs_vec.time_indices.shape[0] == 3
    assert np.array_equal(obs_vec.obs_dims, np.repeat(4, 3))
    assert np.array_equal(obs_vec.location_indices,
                          np.tile([0, 6, 13, 29], 3).reshape((3, 4)))
    assert obs_vec.values[0, 0] == pytest.approx(
            l96.values[3, 0] + obs_vec.errors[0, 0])
    assert np.allclose(obs_vec.values[0],
                       np.array([1.58251908, 1.87502525,
                                 0.48098366, -1.89712787]))
    assert np.allclose(obs_vec.errors[0],
                       np.array([-0.27202828,  0.48749677,
                                 0.5913821, -0.22663247]))
    assert np.allclose(obs_vec.times, np.array([0.15, 0.2, 0.45]))


def test_obs_l96_moving():
    """Tests non-stationary observer for Lorenz96"""
    l96 = data.Lorenz96()
    l96.generate(n_steps=10)

    obs = observer.Observer(
            l96,
            random_time_density=0.4,
            random_location_density=0.3,
            error_sd=1.2,
            stationary_observers=False)
    obs_vec = obs.observe()

    assert obs_vec.values.shape[0] == 3
    assert obs_vec.num_obs == 3
    assert obs_vec.times.shape[0] == 3
    assert obs_vec.time_indices.shape[0] == 3
    assert np.array_equal(obs_vec.obs_dims, np.array([8, 12, 12]))
    assert obs_vec.location_indices[1][-1] == 32
    assert obs_vec.values[0][0] == pytest.approx(
            l96.values[3, 0] + obs_vec.errors[0][0])
    assert obs_vec.values[2][5] == pytest.approx(2.220523949148202)
    assert obs_vec.errors[1][4] == pytest.approx(0.3215829246757838)
    assert np.allclose(obs_vec.times, np.array([0.15, 0.2, 0.45]))


def test_obs_pyqg():
    """Tests observer for PyQG"""
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


@pytest.mark.skip(reason="AWS removed ERA5 data.")
def test_obs_aws():
    """Tests observer for AWS downloaded ERA5 data"""
    aws = data.AWS()
    aws.load()

    obs = observer.Observer(
        aws,
        random_time_density=0.025,
        random_location_density=0.1,
        error_sd=0.0,
        error_bias=5.)
    obs_vec = obs.observe()

    assert obs_vec.values.shape[0] == 219
    assert obs_vec.num_obs == 219
    assert obs_vec.times.shape[0] == 219
    assert obs_vec.time_indices.shape[0] == 219
    assert np.array_equal(obs_vec.obs_dims, np.repeat(58, 219))
    assert obs_vec.location_indices[0, 0] == 17
    assert obs_vec.values[0, 0] == pytest.approx(
            aws.values[72, 17] + 5)
    assert obs_vec.values[123, 42] == pytest.approx(306.0625)
    assert np.array_equal(obs_vec.errors,
                          np.repeat(5, 219*58).reshape(219, 58))
    assert obs_vec.times[1] == np.datetime64('2020-01-04T18:00:00.000000000')


def test_obs_sqgturb():
    """Tests observer for SQGTurb"""
    sqg = data.SQGTurb()
    sqg.generate(n_steps=10)

    obs = observer.Observer(
        sqg,
        random_time_density=0.3,
        random_location_density=0.01,
        error_sd=25.)
    obs_vec = obs.observe()

    assert obs_vec.values.shape[0] == 2
    assert obs_vec.num_obs == 2
    assert obs_vec.times.shape[0] == 2
    assert obs_vec.time_indices.shape[0] == 2
    assert np.array_equal(obs_vec.obs_dims, np.repeat(204, 2))
    assert np.array_equal(obs_vec.location_indices[0, 0],
                          np.array([0, 0, 61]))
    assert obs_vec.values[1, 123] == pytest.approx(
            (sqg.values_gridded[obs_vec.time_indices[1]][
                tuple(obs_vec.location_indices[1, 123])]
             + obs_vec.errors[1, 123]))
    assert obs_vec.values[0, 44] == pytest.approx(2934.9163955098516)
    assert obs_vec.errors[1, 187] == pytest.approx(-16.501806315409056)
    assert np.allclose(obs_vec.times, np.array([2700., 9000.]))
