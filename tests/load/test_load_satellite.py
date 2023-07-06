""" Loading satellite tests

1. Open HRV data
2. Open data
3. Open 15 data
4. Try to open 5, then open 15
5. Test load_and_check_satellite_data, loads data
6. Test load_and_check_satellite_data, no file
7. Test load_and_check_satellite_data, old data


"""
from ocf_datapipes.load import OpenSatellite
from ocf_datapipes.load.satellite import load_and_check_satellite_data
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


def test_open_satellite_5_then_15():
    sat_datapipe = OpenSatellite(
        zarr_path="tests/data/sat_data.zarr", use_15_minute_data_if_needed=True
    )
    metadata = next(iter(sat_datapipe))
    assert metadata is not None


@freeze_time("2020-04-01 14:00:00")
def test_load_and_check_satellite_data():
    dataset, use_15_minute_data = load_and_check_satellite_data(
        zarr_path="tests/data/sat_data.zarr"
    )
    assert dataset is not None
    assert use_15_minute_data is False


def test_load_and_check_satellite_data_no_file():
    dataset, use_15_minute_data = load_and_check_satellite_data(
        zarr_path="tests/data/sat_data_zzzzz.zarr"
    )
    assert dataset is None
    assert use_15_minute_data is True


def test_load_and_check_satellite_old_data():
    dataset, use_15_minute_data = load_and_check_satellite_data(
        zarr_path="tests/data/sat_data.zarr"
    )
    assert dataset is not None
    assert use_15_minute_data is True
