import logging
from timeit import timeit

import numpy as np
import pandas as pd
import xarray as xr

logger = logging.getLogger(__name__)

from ocf_datapipes.select import FilterPVSystemsWithOnlyNanInADay


def test_execution_time():
    time = pd.date_range(start="2022-01-01", end="2023-01-01", freq="5min")
    pv_system_id = [1, 2, 3]
    ALL_COORDS = {"time_utc": time, "pv_system_id": pv_system_id}

    data = np.zeros((len(time), len(pv_system_id)))
    data[:, 2] = np.nan

    data_array = xr.DataArray(
        data,
        coords=ALL_COORDS,
    )
    drop_sys_with_only_nan = FilterPVSystemsWithOnlyNanInADay(
        [data_array], minimum_number_data_points=288
    )
    data_after_drop = next(iter(drop_sys_with_only_nan))
