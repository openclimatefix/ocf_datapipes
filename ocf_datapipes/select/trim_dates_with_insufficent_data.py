"""
This is a class function that slices a contigous datetime range to the 12th hour
"""
import logging

import numpy as np
import xarray as xr
from torchdata.datapipes import functional_datapipe
from torchdata.datapipes.iter import IterDataPipe

logger = logging.getLogger(__name__)


@functional_datapipe("trim_dates_with_insufficent_data")
class TrimDatesWithInsufficentDataIterDataPipe(IterDataPipe):
    """Trim the date values of the Xarray Timeseries data"""

    def __init__(self, source_datapipe: IterDataPipe, minimum_number_data_points: int):
        """Trim the dates to the exact 12th hour

        For the five minute interval, If the time_utc dates are insufficent and
        less than multiple of 289, this method trims that extended dates

        For example:
            <xr.DataArray> time_utc : "2020-01-01T00:00"..."2020-01-02T00:00"..."2020-01-02T06:00"

        This method trims the inusfficent less than one day data at
        the end and provides full set of complete one day interval (5min or 15min or...)

        Args:
            source_datapipe: Xarray emitting timeseries data
            minimum_number_data_points: Minimum number of data intervals in a given day
                5min xarray interval data = 288
                15min xarray interval data = 96
                .........
        """
        self.source_datapipe = source_datapipe
        self.intervals = minimum_number_data_points

    def __iter__(self) -> xr.DataArray():
        # This dropping of insufficent data considers just dates in a given datetime
        for xr_dataset in self.source_datapipe:

            # Getting the 'datetime' values into a single 1D array
            dates_array = xr_dataset.coords["time_utc"].values

            # Checking if the total length of 'datetime' is
            # greater than provided time intervals
            if dates_array.size >= self.intervals:

                # Counting the minute intervals (both non_zero and zero),
                # as every 5min, 15min, or 30 min
                # has intervals such as, for 15 min [0, 15, 30, 45, 0,........]
                total_minute_intervals = xr_dataset.time_utc.dt.minute.values
                count_minute_intervals = np.count_nonzero(
                    total_minute_intervals
                ) + np.count_nonzero(total_minute_intervals == 0)
                # Collecting five minute intervals and counting them
                logger.info(f"Total number of those five minutes are {count_minute_intervals}")

                logger.info(
                    f"Checking if the count is a multiple of given interval {self.intervals}"
                )

                # Checking if the minute intervals are multiples of total intervals in a day
                # For example, datet time values of one day with five minute intervals
                # consists of 288 five-minutes
                check = count_minute_intervals % self.intervals == 0.0

                if not check:
                    # Counting number of intervals needed to be trimmed at the end
                    # The check would be always false, as in a given day,
                    # the last time step would be of the next day
                    trim_dates_position = int(count_minute_intervals % self.intervals)

                    # Number of intervals needed to be trimmed
                    # at the end are trim_dates_position
                    trim_dates = dates_array[-trim_dates_position:]
                    logger.info(f"The trimmed dates are as follows {trim_dates}")

                    # Dropping the dates coordinate variable data and its data in the xarray
                    xr_dataset = xr_dataset.drop_sel(time_utc=trim_dates)

            yield xr_dataset
