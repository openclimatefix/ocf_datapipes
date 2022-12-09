"""Drop PV systems that report overnight"""

import logging
import random
from datetime import datetime
from typing import Dict, List, Union

import numpy as np
import pandas as pd
import xarray as xr
from torchdata.datapipes import functional_datapipe
from torchdata.datapipes.iter import IterDataPipe

logger = logging.getLogger(__name__)

"""According to the UK Met office- https://www.metoffice.gov.uk/weather/learn-about/met-office-for-schools/other-content/other-resources/our-seasons

Summer  - starts in June in the UK and finishes at the end of August.
Autumn -  starts in September and finishes in November
Winter - runs from December to February
Spring - begins in March and ends in May
According to worlddata.info - https://www.worlddata.info/europe/united-kingdom/sunset.php
Month	    Sunrise	    Sunset      Hours of daylight
January	    07:57 am	04:22 pm	8:25 h
February	07:12 am	05:16 pm	10:05 h
March	    06:12 am	06:06 pm	11:54 h
April	    06:02 am	07:58 pm	13:56 h
May	        05:06 am	08:47 pm	15:41 h
June	    04:40 am	09:21 pm	16:41 h
July	    04:59 am	09:13 pm	16:15 h
August	    05:44 am	08:25 pm	14:41 h
September	06:33 am	07:17 pm	12:44 h
October	    07:22 am	06:09 pm	10:47 h
November	07:16 am	04:13 pm	8:57 h
December	07:57 am	03:53 pm	7:56 h
"""


@functional_datapipe("drop_night_pv")
class DropNightPVIterDataPipe(IterDataPipe):
    """Drop the pv output of the over night date times in a timeseries Xarray Dataset.

    This function provides an extra dimension of day and night status for each time step
    in a timeseries Xarray Dataset.With that, it fills NaNs in all the datavariables
    where the status is "night".

    Args:
        source_datapipe: A datapipe that emmits Xarray Dataset of the pv.netcdf file.
    """

    def __init__(self, source_datapipe: IterDataPipe):
        """
        This function provides an extra dimension of day and night status for each time step
        in a timeseries Xarray Dataset

        With that, it fills NaNs in all the datavariables where the status is "night"

        Args
            source_datapipe: A datapipe that emmits Xarray Dataset of the pv.netcdf file
        """
        self.source_datapipe = source_datapipe

    def __iter__(self) -> xr.Dataset():
        logger.warning("This droping of the nighttime pv is only applicable to the UK PV datasets")
        # Classifying the night time
        for xr_dataset in self.source_datapipe:
            # xr_dataset = self.source_datapipe
            dates_list = xr_dataset.coords["datetime"].values.astype("datetime64[s]").tolist()

            uk_daynight_dict = {
                "1": ["17", "7"],
                "2": ["18", "7"],
                "3": ["19", "9"],
                "4": ["20", "6"],
                "5": ["21", "5"],
                "6": ["22", "4"],
                "7": ["22", "5"],
                "8": ["21", "5"],
                "9": ["20", "6"],
                "10": ["21", "7"],
                "11": ["17", "7"],
                "12": ["17", "7"],
            }

            def day_status(date_time_of_day: datetime):
                dtime = date_time_of_day
                date_month = str(dtime.month)
                date_hr = str(dtime.strftime("%H"))
                status = uk_daynight_dict[date_month]
                if date_hr >= status[0] and date_hr <= status[1]:
                    status_day = "night"
                else:
                    status_day = "day"
                return status_day

            status = [day_status(i) for i in dates_list]
            # sanity check
            assert len(dates_list) == len(status)
            xr_dataset = xr_dataset.assign_coords(daynight_status=("datetime", status))
            xr_dataset = xr_dataset.where(xr_dataset.daynight_status == "day")
            yield xr_dataset
