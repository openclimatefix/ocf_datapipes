#!/usr/bin/env python3
# -*- coding: UTF-8 -*-

"""
This is a class function that assigns day night status
"""
import logging

import numpy as np
import xarray as xr
from numba import jit, prange
from torchdata.datapipes import functional_datapipe
from torchdata.datapipes.iter import IterDataPipe

logger = logging.getLogger(__name__)

uk_daynight_dict = {
    1: [7, 17],
    2: [7, 18],
    3: [9, 19],
    4: [6, 20],
    5: [5, 21],
    6: [4, 22],
    7: [5, 22],
    8: [5, 21],
    9: [6, 20],
    10: [7, 21],
    11: [7, 17],
    12: [7, 17],
}
logger.info(
    f"The day and night standard hours are set by {'https://www.timeanddate.com/sun/uk/london'}"
)


@jit(parallel=True)
def add_day_night_status(xr_dataset: xr.DataArray, uk_daynight_dict: dict) -> xr.DataArray:
    """Adds a new dimension of day/night status to the input xarray.DataArray"""

    # Get the month and hour values for each time stamp
    date_month = np.asarray(xr_dataset.time_utc.dt.month.values, dtype=int)
    date_hr = np.asarray(xr_dataset.time_utc.dt.hour.values, dtype=int)
    month_hr_stack = np.stack((date_month, date_hr))

    # Get the day/night status for each time stamp
    status_daynight = []
    for i in prange(len(xr_dataset.coords["time_utc"].values)):
        if month_hr_stack[1][i] in range(
            uk_daynight_dict[month_hr_stack[0][i]][0],
            uk_daynight_dict[month_hr_stack[0][i]][1],
        ):
            status = "day"
        else:
            status = "night"

        status_daynight.append(status)

    # Add the day/night status as a new coordinate
    xr_dataset = xr_dataset.assign_coords(status_daynight=(("time_utc"), status_daynight))

    return xr_dataset


@functional_datapipe("assign_daynight_status")
class AssignDayNightStatusIterDataPipe(IterDataPipe):
    """Adds a new dimension of day/night status"""

    def __init__(self, source_datapipe: IterDataPipe):
        """
        This method adds extra coordinate of day night status.

        Args:
            source_datapipe: Datapipe emiiting Xarray Dataset

        Result:
            <xarray.Dataset>
            Dimensions:   (datetime: 289)
            Coordinates:
            * datetime  (datetime) datetime64[ns]
            * status_day (datetime)
        """

        self.source_datapipe = source_datapipe

    def __iter__(self) -> xr.DataArray():
        """Returns an xarray dataset with extra dimesion"""
        for xr_dataset in self.source_datapipe:
            xr_dataset = add_day_night_status(xr_dataset, uk_daynight_dict)
            yield xr_dataset
