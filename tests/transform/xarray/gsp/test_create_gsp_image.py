import numpy as np

from ocf_datapipes import CreateGSPImage, DropGSP


def test_create_gsp_image(gsp_datapipe, sat_datapipe):
    gsp_datapipe = DropGSP(gsp_datapipe, gsps_to_keep=[0])
    pv_image_datapipe = CreateGSPImage(gsp_datapipe, sat_datapipe)
    data = next(iter(pv_image_datapipe))
    assert np.max(data) > 0
    assert np.min(data) >= 0
