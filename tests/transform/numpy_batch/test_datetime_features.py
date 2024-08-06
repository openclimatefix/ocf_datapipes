import numpy as np
from datetime import datetime

from ocf_datapipes.transform.numpy_batch import AddTrigonometricDateTime

from ocf_datapipes.transform.numpy_batch.datetime_features import _get_date_time_in_pi


def test_get_date_time_in_pi():
    times = [
              "2020-01-01T00:00:01", "2020-04-01T06:00:00",
              "2020-07-01T12:00:00", "2020-09-30T18:00:00",
              "2020-12-31T23:59:59",
              "2021-01-01T00:00:01", "2021-04-02T06:00:00",
              "2021-07-02T12:00:00", "2021-10-01T18:00:00",
              "2021-12-31T23:59:59"
             ]

    expected_times_in_pi = [0, 0.5*np.pi, np.pi, 1.5*np.pi, 2*np.pi]*2

    times = np.array([datetime.fromisoformat(time) for time in times], dtype="datetime64[s]")

    date_in_pi, time_in_pi = _get_date_time_in_pi(times)

    assert np.isclose(np.cos(time_in_pi), np.cos(expected_times_in_pi), atol=1e-04).all()
    assert np.isclose(np.sin(time_in_pi), np.sin(expected_times_in_pi), atol=1e-04).all()
    assert np.isclose(np.cos(date_in_pi), np.cos(expected_times_in_pi), atol=0.01).all()
    assert np.isclose(np.sin(date_in_pi), np.sin(expected_times_in_pi), atol=0.02).all()


def test_add_trigonometric_datetime(combined_datapipe):
    combined_datapipe = AddTrigonometricDateTime(combined_datapipe, modality_name="wind")
    data = next(iter(combined_datapipe))
    assert data is not None
