import logging
from timeit import timeit

import numpy as np
import pandas as pd
import xarray as xr

logger = logging.getLogger(__name__)

from ocf_datapipes.select import DropPVSystemsWithOnlyNanInADay, TrimDatesWithInsufficentData


def test_execution_time():
    time = pd.date_range(start="2022-01-01", end="2023-01-01", freq="5T")
    pv_system_id = [1, 2, 3]
    ALL_COORDS = {"time_utc": time, "pv_system_id": pv_system_id}

    data = np.zeros((len(time), len(pv_system_id)))
    data[:, 2] = np.nan

    data_array = xr.DataArray(
        data,
        coords=ALL_COORDS,
    )
    drop_sys_with_only_nan = DropPVSystemsWithOnlyNanInADay(
        [data_array], minimum_number_data_points=288
    )
    data_after_drop = next(iter(drop_sys_with_only_nan))
    execution_time = timeit(lambda: next(iter((data_after_drop))), number=1)
    logger.info("Testing the execution time for a year of data")
    logger.info(f"Execution time to test:{execution_time:.4f} seconds")
    print(f"Execution time to test for 10k times:{execution_time:.4f} seconds")


def test_trim_12th_hour_timestep(passiv_datapipe):
    data = TrimDatesWithInsufficentData(passiv_datapipe, minimum_number_data_points=288)
    data = next(iter(data))
    count = len(data.coords["time_utc"].values)
    assert count % 288 == 0


def test_trim_dates_lessthan_oneday():
    time = pd.date_range(start="2022-01-01", freq="5T", periods=150)
    pv_system_id = [1, 2, 3]
    ALL_COORDS = {"time_utc": time, "pv_system_id": pv_system_id}

    data = np.zeros((len(time), len(pv_system_id)))
    data[:, 2] = np.nan

    data_array = xr.DataArray(
        data,
        coords=ALL_COORDS,
    )
    trim_dates = TrimDatesWithInsufficentData([data_array], minimum_number_data_points=288)
    data = next(iter(trim_dates))
    assert data.time_utc.values.size == 150


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
    trim_dates = TrimDatesWithInsufficentData([data_array], minimum_number_data_points=288)
    drop_sys_with_only_nan = DropPVSystemsWithOnlyNanInADay(
        trim_dates, minimum_number_data_points=288
    )

    trim_dates = next(iter(trim_dates))
    drop_sys_with_only_nan = next(iter(drop_sys_with_only_nan))

    assert trim_dates.time_utc.values.size == 288
    assert drop_sys_with_only_nan.coords["pv_system_id"].size == 2
