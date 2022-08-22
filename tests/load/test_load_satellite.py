from ocf_datapipes.load import OpenSatellite


def test_open_satellite():
    sat_dp = OpenSatellite(zarr_path="tests/data/hrv_sat_data.zarr")
    metadata = next(iter(sat_dp))
    assert metadata is not None
