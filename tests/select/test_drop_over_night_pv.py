from datetime import datetime
from typing import Optional

import numpy as np
import pandas as pd
import xarray as xr

from ocf_datapipes.select import AssignDayNightStatus, DropNightPV


def test_full_nan_night(passiv_datapipe):
    fill_nan_nights = AssignDayNightStatus(passiv_datapipe)
    data = next(iter(fill_nan_nights))
    data = data[dict(pv_system_id = [0])].values
    # For the month of April(4), according to UK Seasonal cycle <Dict>
    # Hours between 6 and 20 is the day and five minute intervals 
    # between rest of the hours adds upto '120' + 1(for the next day 12th hour)

    # This test counts number of NaN's for a single pv system (in a single day) 
    count = np.count_nonzero(np.isnan(data))
    assert count == 121.

def test_assign_status_night(passiv_datapipe):
    night_status = AssignDayNightStatus(passiv_datapipe, assign_status = True)
    data = next(iter(night_status))
    coords = data.coords["status_day"].values

    # This test follows the same method but counts number of 'night' labels in
    # the newly added 'status_day' coordinate 
    assert np.count_nonzero(coords == "night") == 121.
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
    night_drop_pv = AssignDayNightStatus([data_array], assign_status = True)
    night_drop_pv = DropNightPV(night_drop_pv)

    # check output, has dropped system 3
    data = next(iter(night_drop_pv))
    print(data)
    assert len(data.pv_system_id) == 2
    assert len(data.time_utc) == 289
