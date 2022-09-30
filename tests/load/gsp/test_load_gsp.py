from ocf_datapipes.load import OpenGSP, OpenGSPNational


def test_open_gsp():
    gsp_datapipe = OpenGSP("tests/data/gsp/test.zarr")
    gsp_data = next(iter(gsp_datapipe))
    assert gsp_data is not None


def test_open_gsp_national():
    gsp_datapipe = OpenGSPNational("tests/data/gsp/test.zarr")
    gsp_data = next(iter(gsp_datapipe))

    assert gsp_data.gsp_id.values == 0
    assert gsp_data is not None
