import logging

import numpy as np
import pandas as pd
import xarray as xr

from ocf_datapipes.select import DropPVSysWithNan
from ocf_datapipes.select import TrimDatesWithInsufficentData

def test_with_pvoutput_datapipe(pvoutput_datapipe):
    before_trim_date_and_drop_sys = TrimDatesWithInsufficentData(pvoutput_datapipe, intervals = 288)
    after_trim_date_and_drop_sys = DropPVSysWithNan(before_trim_date_and_drop_sys, intervals= 288)

    before_data = next(iter(before_trim_date_and_drop_sys))
    after_data = next(iter(after_trim_date_and_drop_sys))

    assert len(before_data.coords["time_utc"].values) == len(after_data.coords["time_utc"].values)
    assert len(before_data.coords["pv_system_id"].values) != len(after_data.coords["pv_system_id"].values)

def test_trim_lessthan_oneday(passiv_datapipe):
    data = TrimDatesWithInsufficentData(passiv_datapipe, intervals=288)
    data = next(iter(data))
    count = len(data.coords["time_utc"].values)
    assert count % 288 == 0


def test_sys_without_pv(passiv_datapipe):
    no_pv_wholeday = TrimDatesWithInsufficentData(passiv_datapipe, intervals=288)
    no_pv_wholeday = DropPVSysWithNan(no_pv_wholeday, intervals=288)
    data = next(iter(no_pv_wholeday))
    count = len(data.coords["pv_system_id"].values)
    assert count == 2


def test_constructed_xarray():
    time = pd.date_range(start="2022-01-01", freq="5T", periods=289)
    pv_system_id = [1, 2, 3]
    ALL_COORDS = {"time_utc": time, "pv_system_id": pv_system_id}

    data = np.zeros((len(time), len(pv_system_id)))
    data[:, 2] = np.nan

    data_array = xr.DataArray(
        data,
        coords=ALL_COORDS,
    )
    trim_dates = TrimDatesWithInsufficentData([data_array], intervals=288)
    sys_without_pv = DropPVSysWithNan(trim_dates, intervals = 288)

    data = next(iter(sys_without_pv))

    assert len(data.time_utc.values) == 288
    assert len(data.coords["pv_system_id"]) == 2
