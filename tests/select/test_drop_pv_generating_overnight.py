from datetime import datetime
from typing import Optional

import numpy as np
import pandas as pd
import xarray as xr

from ocf_datapipes.select import DropNightPV
from ocf_datapipes.transform.xarray import AssignDayNightStatus


def test_drop_with_pvoutput_datapipe(pvoutput_datapipe):
    before_dropping_pv_with_night_output = AssignDayNightStatus(pvoutput_datapipe)
    after_dropping_pv_with_night_output = DropNightPV(before_dropping_pv_with_night_output)

    data_before_drop = next(iter(before_dropping_pv_with_night_output))
    data_after_drop = next(iter(after_dropping_pv_with_night_output))

    # In 'pvoutput_datapipe', there are 41 or so systems and some of the systems have been dropped
    # as they generate power over night.
    assert len(data_before_drop.coords["pv_system_id"].values) != len(
        data_after_drop.coords["pv_system_id"].values
    )

def test_drop_with_constructed_dataarray():

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
    before_drop = AssignDayNightStatus([data_array])
    after_drop = DropNightPV(before_drop)

    # check output, has dropped system 3
    before_drop_data = next(iter(before_drop))
    after_drop_data = next(iter(after_drop))

    assert len(before_drop_data.pv_system_id) == 3
    assert len(after_drop_data.pv_system_id) == 2
