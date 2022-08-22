from ocf_datapipes.load import OpenGSP


def test_open_gsp():
    gsp_dp = OpenGSP("tests/data/gsp/test.zarr")
    metadata = next(iter(gsp_dp))
    assert metadata is not None
