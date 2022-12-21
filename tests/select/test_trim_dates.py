import logging

import numpy as np
import pandas as pd
import xarray as xr

from ocf_datapipes.select import TrimDatesWithInsufficentData


def test_trim_lessthan_oneday(passiv_datapipe):
    data = TrimDatesWithInsufficentData(passiv_datapipe, intervals=288)
    data = next(iter(data))
    count = len(data.coords["time_utc"].values)
    assert count % 288 == 0


def test_with_pvoutput_datapipe(pvoutput_datapipe):
    before_trim_date = pvoutput_datapipe
    after_trim_date = TrimDatesWithInsufficentData(before_trim_date, intervals=288)

    before_data = next(iter(before_trim_date))
    after_data = next(iter(after_trim_date))

    assert len(before_data.coords["time_utc"].values) == len(after_data.coords["time_utc"].values)


def test_constructed_xarray():
    time = pd.date_range(start="2022-01-01", freq="5T", periods=350)
    pv_system_id = [1, 2, 3]
    ALL_COORDS = {"time_utc": time, "pv_system_id": pv_system_id}

    data = np.zeros((len(time), len(pv_system_id)))
    data[:, 2] = np.nan

    data_array = xr.DataArray(
        data,
        coords=ALL_COORDS,
    )

    trim_date = TrimDatesWithInsufficentData([data_array], intervals=288)

    data = next(iter(trim_date))

    assert len(data.coords["time_utc"].values) == 288
