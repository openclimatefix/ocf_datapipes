"""Select PV systems and Dates with No output for the entire day"""
import logging
from collections import defaultdict
from datetime import datetime, timedelta
from typing import Dict, List, Union

import numpy as np
import pandas as pd
import xarray as xr
from torchdata.datapipes import functional_datapipe
from torchdata.datapipes.iter import IterDataPipe

from ocf_datapipes.utils.utils import return_sys_indices_which_has_cont_nan as sys_idx_cont_nan

logger = logging.getLogger(__name__)


@functional_datapipe("select_pv_systems_without_output")
class SelectPVSystemsWithoutOutputIterDataPipe(IterDataPipe):
    """Remove any PV systems with less than 1 day of data.

    This is done, by counting all the non values and check the
    count is greater than 289 (number of 5 minute intervals in a day)

        Args:
            source_datapipe: Datapipe of Xarray Dataset emitting timeseries data
    """

    def __init__(self, source_datapipe: IterDataPipe, intervals: int) -> None:

        self.source_datapipe = source_datapipe
        self.intervals = intervals

    def __iter__(self) -> xr.Dataset():

        logger.info(f"\nReading the DataArray\n")
        for xr_dataset in self.source_datapipe:

            dates_array = xr_dataset.coords["time_utc"].values
            logger.info(
                f"\nCollecting the time series data from 'time_utc' coordinate\n {dates_array}\n"
            )

            for i in range(0, len(dates_array), self.intervals):
                logger.info(
                    f"\nGetting the first and last timestamp of the day based on {self.intervals} intervals\n"
                )

                logger.info(
                    f"\nSlicing the dataset by individual first timestamp in a day {dates_array[i]}\n"
                )
                logger.info(
                    f"\nand last timestamp of the day {dates_array[i+(self.intervals-1)]}\n"
                )
                xr_ds = xr_dataset.sel(
                    time_utc=slice(dates_array[i], dates_array[i + (self.intervals - 1)])
                )

                logger.info(f"\nExtracting system ids with continous NaN's\n")
                sys_ids = sys_idx_cont_nan(xr_ds.values, check_interval=self.intervals)

                logger.warning(f"\nDropping the systems which are inactive for the whole day\n")
                if not len(sys_ids) == 0:
                    xr_dataset = xr_dataset.drop_isel(pv_system_id=sysids)

            yield xr_dataset
