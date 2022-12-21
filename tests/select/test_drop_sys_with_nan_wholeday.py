import logging

import numpy as np
import pandas as pd
import xarray as xr

from ocf_datapipes.select import DropPVSysWithNan


def test_with_pvoutput_datapipe(pvoutput_datapipe):
    before_drop_pv_sys_with_nan_wholeday = pvoutput_datapipe
    after_drop_pv_sys_with_nan_wholeday = DropPVSysWithNan(
        before_drop_pv_sys_with_nan_wholeday, intervals=288
    )

    before_data = next(iter(before_drop_pv_sys_with_nan_wholeday))
    after_data = next(iter(after_drop_pv_sys_with_nan_wholeday))

    assert len(before_data.coords["pv_system_id"].values) != len(
        after_data.coords["pv_system_id"].values
    )


def test_sys_without_pv(passiv_datapipe):
    drop_pv_sys_with_nan_wholeday = DropPVSysWithNan(passiv_datapipe, intervals=288)
    data = next(iter(drop_pv_sys_with_nan_wholeday))
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

    drop_pv_sys_with_nan_wholeday = DropPVSysWithNan([data_array], intervals=288)

    data = next(iter(drop_pv_sys_with_nan_wholeday))

    assert len(data.coords["pv_system_id"].values) == 2
