import pandas as pd
import numpy as np
from datetime import datetime

from ocf_datapipes.transform.numpy_batch import AddSunPosition

from ocf_datapipes.transform.numpy_batch.sun_position import _get_azimuth_and_elevation


def test_get_azimuth_and_elevation():
    lon = -0.8
    lat = 51.64
    times = ["2024-01-31T12:30:00", "2024-01-31T13:00:00"]
    times = np.array([datetime.fromisoformat(time) for time in times], dtype="datetime64[s]")

    azimuth, elevation = _get_azimuth_and_elevation(lon=lon, lat=lat, dt=times, must_be_finite=True)
    assert elevation.mean() > 20
    assert elevation.mean() < 21
    assert azimuth.mean() > 180
    assert azimuth.mean() < 200


def test_add_sun_position_pv(combined_datapipe):
    combined_datapipe = AddSunPosition(combined_datapipe, modality_name="pv")
    data = next(iter(combined_datapipe))


def test_add_sun_position_gsp(combined_datapipe):
    combined_datapipe = AddSunPosition(combined_datapipe, modality_name="gsp")
    data = next(iter(combined_datapipe))
