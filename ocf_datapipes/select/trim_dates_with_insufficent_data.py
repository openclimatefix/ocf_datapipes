import logging
from datetime import datetime

import numpy as np
import xarray as xr
from torchdata.datapipes import functional_datapipe
from torchdata.datapipes.iter import IterDataPipe

logger = logging.getLogger(__name__)


@functional_datapipe("trim_dates_with_insufficent_data")
class TrimDatesWithInsufficentDataIterDataPipe(IterDataPipe):
    """Trim the date values of the Xarray Timeseries data"""

    def __init__(self, source_datapipe: IterDataPipe, intervals: int):
        """
        For the five minute interval, If the time_utc dates are insufficent and
        less than multiple of 289, this method trims that extended dates

        For example:
            <xr.DataArray> time_utc : "2020-01-01T00:00"........"2020-01-02T00:00"....."2020-01-02T06:00"

        This method trims the inusfficent less than one day data at the end and provides full set of complete one day
        interval (5min or 15min or........)

        Args:
            source_datapipe: Xarray emitting timeseries data
            intervals:
                5min xarray interval data = 288
                15min xarray interval data = 96
                .........
        """
        self.source_datapipe = source_datapipe
        self.intervals = intervals

    def __iter__(self) -> xr.DataArray():
        logger.info(
            f"\nThis dropping of insufficent data considers just dates in a given datetime\n"
        )
        for xr_dataset in self.source_datapipe:

            total_five_minutes = np.asarray(xr_dataset.time_utc.dt.minute.values, dtype=int)
            count_five_minutes = np.count_nonzero(total_five_minutes) + np.count_nonzero(
                total_five_minutes == 0
            )
            logger.info(f"\nCollecting five minute intervals and counting them\n")
            logger.info(
                f"\nTotal five minute intervals are {total_five_minutes}\nand number of those five minutes are {count_five_minutes}\n"
            )

            time_series = np.asarray(xr_dataset.coords["time_utc"].values)
            logger.info(
                f"\nCollecting the time series data from the xarray 'time_utc' coordinate {time_series}\n"
            )

            logger.info(
                f"\nChecking if the count is a multiple of given interval {self.intervals}\n"
            )
            check = count_five_minutes % self.intervals == 0.0

            if check == False:
                logger.info(f"\nCounting number of intervals needed to be trimmed at the end\n")

                # The check would be always false, as in a given day,
                # the last time step would be of the next day
                trim_dates_position = int(count_five_minutes % self.intervals)
                logger.info(
                    f"\nNumber of intervals needed to be trimmed at the end are {trim_dates_position}\n"
                )

                trim_dates = time_series[-trim_dates_position:]
                logger.info(f"\nThe trimmed dates are as follows {trim_dates}\n")

                logger.warning(
                    f"\nDropping the dates coordinate variable data and its data in the xarray\n"
                )
                new_xr_dataset = xr_dataset.drop_sel(time_utc=trim_dates)

                yield new_xr_dataset
