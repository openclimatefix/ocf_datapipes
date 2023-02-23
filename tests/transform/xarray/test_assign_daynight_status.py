import logging
from timeit import timeit

import numpy as np
import pandas as pd
import xarray as xr

from ocf_datapipes.transform.xarray import AssignDayNightStatus

logger = logging.getLogger(__name__)


def test_assign_status_night(passiv_datapipe):
    night_status = AssignDayNightStatus(passiv_datapipe)
    data = next(iter(night_status))
    data_coords = data.coords["status_daynight"].values
    # For the fourth month in the passiv_datapipe, according to
    # the uk daynight dictionary set in the assign_daynight_status.py
    # total night status count of 5 minute timseries data is 121
    assert np.count_nonzero(data_coords == "night") == 121.0
    assert "day" and "night" in data.coords["status_daynight"].values


def test_time(passiv_datapipe):
    # Create the instance of the AssignDayNightStatusIterDataPipe class
    datapipe = AssignDayNightStatus(passiv_datapipe)

    # Using timeit to measure the execution time of the __iter__ method
    # number of simulations takes place are 365 (meaning for a year),
    # number of simulations takes place are 365 (meaning for a year),
    # as the datapipe considers one day worth of data
    execution_time = timeit(lambda: next(iter((datapipe))), number=365)

    logger.info(f"Execution time to test for 365 times:\n{execution_time:.4f} seconds")
    print(f"\nExecution time to test for 365 times: {execution_time:.4f} seconds")


def test_with_constructed_array():
    time = pd.date_range("2022-01-01 17:00", "2022-01-01 23:55", freq="5min")
    pv_system_id = [1, 2, 3]
    ALL_COORDS = {"time_utc": time, "pv_system_id": pv_system_id}

    data = np.zeros((len(time), len(pv_system_id)))
    data[:, 2] = 1.0

    data_array = xr.DataArray(
        data,
        coords=ALL_COORDS,
    )

    night_status = AssignDayNightStatus([data_array])
    data = next(iter(night_status))
    data_coords = data.coords["status_daynight"].values

    # As the time range only includes night timestamps
    # status "day" is not assigned to any of the timestamps
    assert "day" not in data_coords
