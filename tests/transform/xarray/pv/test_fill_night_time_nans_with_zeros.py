import numpy as np

from ocf_datapipes.transform.xarray import PVFillNightNans


def test_pv_power_remove_data(passiv_datapipe):
    data_before = next(iter(passiv_datapipe))
    
    passiv_datapipe = PVFillNightNans(passiv_datapipe)
    data_after = next(iter(passiv_datapipe))
    
    # status_daynight is added when the night time values are filled
    is_night = data_after.status_daynight=='night'
    
    # Make sure the input data has some night-time NaNs
    assert data_before.where(is_night, drop=True).isnull().any()

    # Make sure the NaNs are gone
    assert not data_after.where(is_night, drop=True).isnull().any()

    # Make sure the day-time values are uneffected
    assert data_before.where(~is_night, drop=True).identical(data_after.where(~is_night, drop=True))