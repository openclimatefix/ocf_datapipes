import numpy as np

from ocf_datapipes.transform.xarray import CreatePVMetadataImage


def test_create_pv_meta_image(passiv_datapipe, sat_datapipe):
    pv_image_datapipe = CreatePVMetadataImage(passiv_datapipe, sat_datapipe)
    data = next(iter(pv_image_datapipe))
    assert np.max(data) > 0


def test_create_pv_meta_image_normalized(passiv_datapipe, sat_datapipe):
    pv_image_datapipe = CreatePVMetadataImage(passiv_datapipe, sat_datapipe, normalize=True)
    data = next(iter(pv_image_datapipe))
    assert np.max(data) > 0
    assert np.isclose(np.min(data), 0.0)

def test_create_pv_meta_image_pvoutput(pvoutput_datapipe, sat_datapipe):
    pv_image_datapipe = CreatePVMetadataImage(pvoutput_datapipe, sat_datapipe)
    data = next(iter(pv_image_datapipe))
    assert np.max(data) > 0


def test_create_pv_meta_image_normalized_pvoutput(pvoutput_datapipe, sat_datapipe):
    pv_image_datapipe = CreatePVMetadataImage(pvoutput_datapipe, sat_datapipe, normalize=True)
    data = next(iter(pv_image_datapipe))
    assert np.max(data) > 0
    assert np.isclose(np.min(data), 0.0)