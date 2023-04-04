import numpy as np

from ocf_datapipes.transform.xarray import CreateTimeImage


def test_create_time_image_sat(sat_datapipe):
    time_image_datapipe = CreateTimeImage(sat_datapipe)
    data = next(iter(time_image_datapipe))
    assert 1.0 >= np.max(data)
    assert -1.0 <= np.min(data)


def test_create_time_image_hrv(sat_hrv_datapipe):
    time_image_datapipe = CreateTimeImage(sat_hrv_datapipe)
    data = next(iter(time_image_datapipe))
    assert 1.0 >= np.max(data)
    assert -1.0 <= np.min(data)
