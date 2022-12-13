
from ocf_datapipes.transform.xarray import EnsureNGSPSPerExampleIter


def test_create_gsp_image_expand(gsp_datapipe):
    gsp_datapipe = EnsureNGSPSPerExampleIter(gsp_datapipe, n_gsps_per_example=123)
    data = next(iter(gsp_datapipe))
    assert len(data.gsp_id) == 123


def test_create_gsp_image_reduce(gsp_datapipe):
    gsp_datapipe = EnsureNGSPSPerExampleIter(gsp_datapipe, n_gsps_per_example=2)
    data = next(iter(gsp_datapipe))
    assert len(data.gsp_id) == 2

