from datetime import datetime
from typing import Optional

import numpy as np
import pandas as pd
import xarray as xr

from ocf_datapipes.select import DropNightPV
from ocf_datapipes.transform.xarray import AssignDayNightStatus

# TODO Tests with PVOuptput_datapipe


def test_assign_status_night(passiv_datapipe):
    night_status = AssignDayNightStatus(passiv_datapipe)
    data = next(iter(night_status))
    coords = data.coords["status_day"].values
    assert np.count_nonzero(coords == "night") == 121.0
    assert "day" and "night" in data.coords["status_day"].values


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
    night_drop_pv = AssignDayNightStatus([data_array])
    night_drop_pv = DropNightPV(night_drop_pv)

    # check output, has dropped system 3
    data = next(iter(night_drop_pv))
    assert len(data.pv_system_id) == 2
    assert len(data.time_utc) == 289
