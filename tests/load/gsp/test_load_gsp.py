from ocf_datapipes.load import OpenGSP


def test_open_gsp():
    gsp_datapipe = OpenGSP("/home/jacob/Development/ocf_datapipes/tests/data/gsp/test.zarr")
    metadata = next(iter(gsp_datapipe))
    assert metadata is not None
