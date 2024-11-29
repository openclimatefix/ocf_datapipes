""" Loading satellite tests

1. Open HRV data
2. Open data
3. Open 15 data

"""

from ocf_datapipes.load import open_sat_data
from freezegun import freeze_time


def test_open_satellite():
    sat_xr = open_sat_data(zarr_path="tests/data/hrv_sat_data.zarr")
    assert sat_xr is not None


def test_open_hrvsatellite():
    sat_xr = open_sat_data(zarr_path="tests/data/sat_data.zarr")
    assert sat_xr is not None


def test_open_satellite_15():
    sat_xr = open_sat_data(zarr_path="tests/data/sat_data_15.zarr")
    assert sat_xr is not None
