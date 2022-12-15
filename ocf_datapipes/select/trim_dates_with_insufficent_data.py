import logging
from datetime import datetime

import numpy as np
import xarray as xr
from torchdata.datapipes import functional_datapipe
from torchdata.datapipes.iter import IterDataPipe

logger = logging.getLogger(__name__)


@functional_datapipe("drop_systems_with_lessthan_oneday_data")
class TrimDatesWithInsufficentDataIterDataPipe(IterDataPipe):
    """Trim the date values of the Xarray Timeseries data"""

    def __init__(self, source_datapipe: IterDataPipe):
        """
        For the five minute interval, If the time_utc dates are insufficent and
        less than multiple of 289, this method trims that extended dates

        For example:
            <xr.DataArray> time_utc : "2020-01-01T00:00"........"2020-01-02T00:00"....."2020-01-02T06:00"

        This method trims the inusfficent less than one day data at the end and provides full set of complete one day
        data intervals
        """
        self.source_datapipe = source_datapipe

    def __iter__(self) -> xr.Dataset():

        for xr_dataset in self.source_datapipe:

            logger.warning(
                f"This dropping of insufficent data considers just dates in a given datetime"
            )
            only_dates = np.asarray(xr_dataset.time_utc.dt.day.values, dtype=int)
            xr_dataset = xr_dataset.assign_coords(only_dates=(("time_utc"), only_dates))
            dates_groups = xr_dataset.groupby("only_dates").groups
            dates, counts = np.unique(only_dates, return_counts=True)
            drop_dates = []
            for idx in range(len(counts)):
                check = counts[idx] == 288.0
                if not check:
                    drop_dates.append(dates[idx])

            for dates in drop_dates:
                xr_dataset = xr_dataset.where(xr_dataset.only_dates != dates, drop=True)
            xr_dataset = xr_dataset.reset_coords("only_dates", drop=True)
            yield xr_dataset
