import logging
from timeit import timeit

import numpy as np
import pandas as pd
import xarray as xr

from ocf_datapipes.select import FilterPvSysGeneratingOvernight as DropNightPV
from ocf_datapipes.transform.xarray import AssignDayNightStatus

logger = logging.getLogger(__name__)


def test_drop_with_passiv_datapipe(passiv_datapipe):
    before_dropping_pv_with_night_output = AssignDayNightStatus(passiv_datapipe)
    after_dropping_pv_with_night_output = DropNightPV(
        before_dropping_pv_with_night_output, threshold=0
    )

    data_before_drop = next(iter(before_dropping_pv_with_night_output))
    data_after_drop = next(iter(after_dropping_pv_with_night_output))
    after_id = data_after_drop.coords["pv_system_id"].values

    Num_of_sys_before_drop = len(data_before_drop.coords["pv_system_id"].values)
    Num_of_sys_after_drop = len(data_after_drop.coords["pv_system_id"].values)

    # In 'pvoutput_datapipe', there are 41 or so systems and some of the systems have been dropped
    # as they generate power over night.
    logger.info("Test1")
    logger.info(f"For the {'pvoutput_datapipe'}")
    logger.info(f"Number of systems before dropping are {Num_of_sys_before_drop}")
    logger.info(f"Number of remaining systems after dropping are {Num_of_sys_after_drop}")
    assert Num_of_sys_before_drop != Num_of_sys_after_drop


def test_time(passiv_datapipe):
    # Create the instance of the AssignDayNightStatusIterDataPipe class
    before_dropping_pv_with_night_output = AssignDayNightStatus(passiv_datapipe)
    after_dropping_pv_with_night_output = DropNightPV(before_dropping_pv_with_night_output)
    data_after_drop = next(iter(after_dropping_pv_with_night_output))

    # Using timeit to measure the execution time of the __iter__ method
    # The number has been chnaged to 10,000 which means running the loop
    # 10k times
    execution_time = timeit(lambda: next(iter((data_after_drop))), number=10000)
    logger.info("Test2")
    logger.info(f"Execution time to test for 10k times:\n{execution_time:.4f} seconds")


def test_drop_with_constructed_dataarray():
    # Make 3 pv systems
    # 1 and 2, produce no power in the night
    # 3 produces power in the night

    time = pd.date_range(start="2022-01-01", freq="5T", periods=289)
    pv_system_id = [1, 2, 3]
    ALL_COORDS = {"time_utc": time, "pv_system_id": pv_system_id}

    # This data array has three systems with comination of pv outputs
    # sys1 = combination of [np.nan,...... 0.,....]
    # sys2 = just zeros [0., 0., 0., ....]
    # sys3 = combination of [np.nan,...... 1.,.....]
    data = np.zeros((len(time), len(pv_system_id)))
    data[:, 2] = 1.0
    data[: data.shape[0] // 2, 0] = np.nan
    data[: data.shape[0] // 2, 2] = np.nan

    data_array = xr.DataArray(
        data,
        coords=ALL_COORDS,
    )
    data_array = data_array.assign_coords(
        observed_capacity_wp=("pv_system_id", np.ones(len(pv_system_id))),
    )

    # run the function
    # Drop Night PV function drops only the third system (sys3)
    before_drop = AssignDayNightStatus([data_array])
    after_drop = DropNightPV(before_drop)

    # check output, has dropped system 3
    before_drop_data = next(iter(before_drop))
    after_drop_data = next(iter(after_drop))
    logger.info("Test3")
    logger.info(f"Remaining systems are dropping are {after_drop_data.pv_system_id.values}")
    assert len(before_drop_data.pv_system_id) == 3
    assert len(after_drop_data.pv_system_id) == 2
