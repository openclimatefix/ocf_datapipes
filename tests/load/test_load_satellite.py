""" Loading satellite tests

1. Open HRV data
2. Open data
3. Open 15 data

"""

from ocf_datapipes.load import OpenSatellite
from freezegun import freeze_time


def test_open_satellite():
    sat_datapipe = OpenSatellite(zarr_path="tests/data/hrv_sat_data.zarr")
    metadata = next(iter(sat_datapipe))
    assert metadata is not None


def test_open_hrvsatellite():
    sat_datapipe = OpenSatellite(zarr_path="tests/data/sat_data.zarr")
    metadata = next(iter(sat_datapipe))
    assert metadata is not None


def test_open_satellite_15():
    sat_datapipe = OpenSatellite(zarr_path="tests/data/sat_data_15.zarr")
    metadata = next(iter(sat_datapipe))
    assert metadata is not None
