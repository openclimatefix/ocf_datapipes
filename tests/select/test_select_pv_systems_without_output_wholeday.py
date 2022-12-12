import logging

import numpy as np
import pandas as pd
import xarray as xr

from ocf_datapipes.select import DropLessThanOneDay, RemoveBadSystems
from ocf_datapipes.select import SelectSysWithoutOutputWholeday as SysWithoutPV


# TODO Still needed to work on this tests a bit
def test_sys_with_lessthan_oneday_pv():
    time = pd.date_range(start="2022-01-01", freq="5T", periods=350)
    data_array = np.stack((np.random.rand(350), np.tile(np.nan, 350)))
    sysid = ["10003", "10004"]

    xr_dict = {
        "coords": {"time_utc": {"dims": "time_utc", "data": time}},
        "dims": "time_utc",
        "data_vars": {
            sysid[0]: {"dims": "time_utc", "data": data_array[0]},
            sysid[1]: {"dims": "time_utc", "data": data_array[1]},
        },
    }
    xr_dataset = xr.DataArray.from_dict(xr_dict)

    data = DropLessThanOneDay(xr_dataset)
    data = next(iter(data))
    assert len(list(data.keys())) == 0.0


def test_sys_without_pv(passiv_datapipe):
    no_pv_wholeday = SysWithoutPV(passiv_datapipe)
    data = next(iter(no_pv_wholeday))
    key1 = sorted(data.keys())
    key2 = sorted(list({k2 for v in data.values() for k2 in v}))
    for x in key1:
        for y in key2:
            assert data[x][y] == "Active" or "Inactive"


def test_sys_without_passivpv(passiv_datapipe):
    no_pv_wholeday = RemoveBadSystems(passiv_datapipe)
    data = next(iter(no_pv_wholeday))
    assert len(list(data)) != 0
