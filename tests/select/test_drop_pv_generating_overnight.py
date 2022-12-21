from datetime import datetime
from typing import Optional

import numpy as np
import pandas as pd
import xarray as xr

from ocf_datapipes.select import DropNightPV
from ocf_datapipes.transform.xarray import AssignDayNightStatus


def test_with_pvoutput_datapipe(pvoutput_datapipe):
    before_dropping_pv_with_night_output = AssignDayNightStatus(pvoutput_datapipe)
    after_dropping_pv_with_night_output = DropNightPV(before_dropping_pv_with_night_output)

    data_before_drop = next(iter(before_dropping_pv_with_night_output))
    data_after_drop = next(iter(after_dropping_pv_with_night_output))

    assign_status_before_drop = data_before_drop.coords["status_day"].values
    assign_status_after_drop = data_after_drop.coords["status_day"].values

    assert len(data_before_drop.coords["pv_system_id"].values) != len(
        data_after_drop.coords["pv_system_id"].values
    )

    assert "day" and "night" in assign_status_before_drop
    assert "day" in assign_status_after_drop


def test_drop_overnight_pvoutput_datapipe(pvoutput_datapipe):
    night_status = AssignDayNightStatus(pvoutput_datapipe)
    drop_night_pv = DropNightPV(night_status)
    data = next(iter(drop_night_pv))
    coords = data.coords["status_day"].values
    assert "day" in coords


def test_drop_night_pv_one_system_power_overnight():

    # Make 3 pv systems
    # 1 and 2, produce no power in the night
    # 3 produces power in the night

    time = pd.date_range(start="2022-01-01", freq="5T", periods=289)
    pv_system_id = [1, 2, 3]
    ALL_COORDS = {"time_utc": time, "pv_system_id": pv_system_id}

    data = np.zeros((len(time), len(pv_system_id)))
    data[:, 2] = 1.0

    data_array = xr.DataArray(
        data,
        coords=ALL_COORDS,
    )

    # run the function
    assign_daynight = AssignDayNightStatus([data_array])
    night_drop_pv = DropNightPV(assign_daynight)

    # check output, has dropped system 3
    data = next(iter(night_drop_pv))
    assert len(data.pv_system_id) == 2
    assert len(data.time_utc) == 289
