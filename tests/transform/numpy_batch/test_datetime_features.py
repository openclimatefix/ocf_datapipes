import numpy as np

from ocf_datapipes.transform.numpy_batch.datetime_features import _get_date_time_in_pi


def test_get_date_time_in_pi():
    times = np.array([
              "2020-01-01T00:00:00", "2020-04-01T06:00:00",
              "2020-07-01T12:00:00", "2020-09-30T18:00:00",
              "2020-12-31T23:59:59",
              "2021-01-01T00:00:00", "2021-04-02T06:00:00",
              "2021-07-02T12:00:00", "2021-10-01T18:00:00",
              "2021-12-31T23:59:59"
             ]).reshape((2, 5))

    expected_times_in_pi = np.array([0, 0.5*np.pi, np.pi, 1.5*np.pi, 2*np.pi] * 2).reshape((2,5))

    times = times.astype("datetime64[s]")

    date_in_pi, time_in_pi = _get_date_time_in_pi(times)

    # Note on precision: times are compared with tolerance equivalent to 1 second,
    # dates are compared with tolerance equivalent to 5 minutes
    # None of the data we use has a higher time resolution, so this is a good test of
    # whether not accounting for leap years breaks things
    assert np.isclose(np.cos(time_in_pi), np.cos(expected_times_in_pi), atol=7.3e-05).all()
    assert np.isclose(np.sin(time_in_pi), np.sin(expected_times_in_pi), atol=7.3e-05).all()
    assert np.isclose(np.cos(date_in_pi), np.cos(expected_times_in_pi), atol=0.02182).all()
    assert np.isclose(np.sin(date_in_pi), np.sin(expected_times_in_pi), atol=0.02182).all()

    # 1D array test
    assert np.isclose(np.cos(time_in_pi[0]), np.cos(expected_times_in_pi[0]), atol=7.3e-05).all()
    assert np.isclose(np.sin(time_in_pi[0]), np.sin(expected_times_in_pi[0]), atol=7.3e-05).all()
    assert np.isclose(np.cos(date_in_pi[0]), np.cos(expected_times_in_pi[0]), atol=0.02182).all()
    assert np.isclose(np.sin(date_in_pi[0]), np.sin(expected_times_in_pi[0]), atol=0.02182).all()
