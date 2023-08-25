from ocf_datapipes.transform.xarray import StackXarray
from ocf_datapipes.select import LocationPicker
import numpy as np


def test_stack_xarray(sat_hrv_datapipe, sat_datapipe, nwp_datapipe, passiv_datapipe):
    loc1, loc2 = LocationPicker(passiv_datapipe).fork(2)
    sat_hrv_datapipe = sat_hrv_datapipe.select_spatial_slice_pixels(
        loc1,
        roi_height_pixels=32,
        roi_width_pixels=32,
    )
    sat_datapipe = sat_datapipe.select_spatial_slice_pixels(
        loc2,
        roi_height_pixels=32,
        roi_width_pixels=32,
    )
    datapipe = StackXarray([sat_hrv_datapipe, sat_datapipe])
    data = next(iter(datapipe))
    assert isinstance(data, np.ndarray)
    assert not np.isnan(data).any()
