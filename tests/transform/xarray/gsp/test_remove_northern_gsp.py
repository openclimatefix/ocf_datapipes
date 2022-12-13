from ocf_datapipes.transform.xarray import RemoveNorthernGSP


def test_remove_northern_gsp(gsp_datapipe):
    gsp_datapipe = RemoveNorthernGSP(gsp_datapipe)
    data = next(iter(gsp_datapipe))

    assert len(data.gsp_id) == 22


def test_remove_northern_gsp_all(gsp_datapipe):
    gsp_datapipe = RemoveNorthernGSP(gsp_datapipe, northern_y_osgb_limit=0)
    data = next(iter(gsp_datapipe))

    assert len(data.gsp_id) == 0


def test_remove_northern_gsp_some(gsp_datapipe):
    northern_y_osgb_limit = 180000

    gsp_datapipe = RemoveNorthernGSP(gsp_datapipe, northern_y_osgb_limit=northern_y_osgb_limit)
    data = next(iter(gsp_datapipe))

    assert len(data.gsp_id) == 5
    assert (data.y_osgb < northern_y_osgb_limit).all()
