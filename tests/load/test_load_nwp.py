from ocf_datapipes.load import OpenNWP

def test_load_nwp():
    nwp_dp = OpenNWP(zarr_path="tests/data/nwp_data/test.zarr")
    metadata = next(iter(nwp_dp))
    assert metadata is not None
