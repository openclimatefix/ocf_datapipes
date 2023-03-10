import numpy as np

from ocf_datapipes.transform.xarray import CreatePVImage


def test_create_pv_image(passiv_datapipe, sat_datapipe):
    pv_image_datapipe = CreatePVImage(passiv_datapipe, sat_datapipe)
    data = next(iter(pv_image_datapipe))
    assert np.max(data) > 0


def test_create_pv_image_take_last_value(passiv_datapipe, sat_datapipe):
    pv_image_datapipe = CreatePVImage(passiv_datapipe, sat_datapipe, take_n_pv_values_per_pixel=1)
    data = next(iter(pv_image_datapipe))
    assert np.max(data) > 0


def test_create_pv_image_normalized(passiv_datapipe, sat_datapipe):
    pv_image_datapipe = CreatePVImage(passiv_datapipe, sat_datapipe, normalize=True)
    data = next(iter(pv_image_datapipe))
    assert np.isclose(np.max(data), 1.0)
    assert np.isclose(np.min(data), 0.0)


def test_create_pv_image_normalized_pvlib(passiv_datapipe, sat_datapipe):
    pv_image_datapipe = CreatePVImage(passiv_datapipe, sat_datapipe, normalize_by_pvlib=True)
    data = next(iter(pv_image_datapipe))
    assert np.max(data) > np.min(data)
    assert np.isclose(np.min(data), 0.0)

def test_create_pv_image_pvoutput(pvoutput_datapipe, sat_datapipe):
    pv_image_datapipe = CreatePVImage(pvoutput_datapipe, sat_datapipe)
    data = next(iter(pv_image_datapipe))
    assert np.max(data) > 0


def test_create_pv_image_take_last_value_pvoutput(pvoutput_datapipe, sat_datapipe):
    pv_image_datapipe = CreatePVImage(pvoutput_datapipe, sat_datapipe, take_n_pv_values_per_pixel=1)
    data = next(iter(pv_image_datapipe))
    assert np.max(data) > 0


def test_create_pv_image_normalized_pvoutput(pvoutput_datapipe, sat_datapipe):
    pv_image_datapipe = CreatePVImage(pvoutput_datapipe, sat_datapipe, normalize=True)
    data = next(iter(pv_image_datapipe))
    assert np.isclose(np.max(data), 1.0)
    assert np.isclose(np.min(data), 0.0)


def test_create_pv_image_normalized_pvlib_pvoutput(pvoutput_datapipe, sat_datapipe):
    pv_image_datapipe = CreatePVImage(pvoutput_datapipe, sat_datapipe, normalize_by_pvlib=True)
    data = next(iter(pv_image_datapipe))
    assert np.max(data) > np.min(data)
    assert np.isclose(np.min(data), 0.0)
