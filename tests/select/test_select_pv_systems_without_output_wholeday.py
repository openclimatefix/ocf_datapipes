import logging

import numpy as np
import pandas as pd
import xarray as xr

from ocf_datapipes.select import RemoveBadSystems
from ocf_datapipes.select import SelectSysWithoutOutputWholeday as SysWithoutPV
from ocf_datapipes.select import TrimDatesWithInsufficentData

def test_trim_lessthan_oneday(passiv_datapipe):
    data = TrimDatesWithInsufficentData(passiv_datapipe, intervals=288)
    data = next(iter(data))
    count = len(data.coords["time_utc"].values)
    assert count % 288.0 == 0.0


def test_sys_without_pv(passiv_datapipe):
    no_pv_wholeday = TrimDatesWithInsufficentData(passiv_datapipe, intervals=288)
    no_pv_wholeday = SysWithoutPV(no_pv_wholeday, intervals=288)
    data = next(iter(no_pv_wholeday))
    count = len(data.coords["pv_system_id"].values)
    assert count == 2.0


# def test_constructed_xarray():
#     time = pd.date_range(start="2022-01-01", freq="5T", periods=350)
#     arr1 = np.random.rand(350)
#     arr2 = np.tile(np.nan, 350)
#     data_array = np.transpose([arr1, arr2])

#     ds = xr.DataArray(
#         data=data_array,
#         dims=["time_utc", "pv_system_id"],
#         coords=dict(pv_system_id=(["pv_system_id"], ["ID1", "ID2"]), time_utc=(["time_utc"], time)),
#     )

#     trim_dates = TrimDatesWithInsufficentData(ds, intervals=288)
#     # sys_without_pv = SysWithoutPV(trim_dates, intervals = 288)

#     sys_without_pv = next(iter(trim_dates))

#     assert len(trim_dates.time_utc.values) == 288.0
#     # assert len(test_sys_without_pv.coords["pv_system_id"]) == 1.0
