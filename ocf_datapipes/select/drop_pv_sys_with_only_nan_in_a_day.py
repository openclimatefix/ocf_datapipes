"""
Drop PV systems which has only NaN's in a single day
"""
import logging

import xarray as xr
from torchdata.datapipes import functional_datapipe
from torchdata.datapipes.iter import IterDataPipe

from ocf_datapipes.utils.utils import return_system_indices_which_has_contiguous_nan

logger = logging.getLogger(__name__)


@functional_datapipe("drop_pv_systems_with_only_nan_in_a_day")
class DropPVSystemsWithOnlyNanInADayIterDataPipe(IterDataPipe):
    """Remove any PV systems with less than 1 day of data"""

    def __init__(self, source_datapipe: IterDataPipe, minimum_number_data_points: int) -> None:
        """Remove any PV systems with less than 1 day of data

        This is done, by counting all the non values and check the
        count is greater than 289 (number of 5 minute intervals in a day)

        Args:
            source_datapipe: Datapipe of Xarray Dataset emitting timeseries data
            minimum_number_data_points: Minimum number of intervals in a given day
                For 5min time series, intervals = 288
                For 30min time series, intervals = 48
        """

        self.source_datapipe = source_datapipe
        self.intervals = minimum_number_data_points

    def __iter__(self) -> xr.Dataset():

        # Reading the DataArray
        for xr_dataset in self.source_datapipe:

            dates_array = xr_dataset.coords["time_utc"].values

            # Checking length of time series longer than standard intervals
            if dates_array.size >= self.intervals:

                # Collecting the time series data from 'time_utc' coordinate
                for i in range(0, dates_array.size, self.intervals):

                    if i == self.intervals:
                        break
                    else:

                        # Getting the first and last timestamp of the day based on intervals
                        # Slicing the dataset by individual first timestamp in a day
                        # and last timestamp of the day
                        xr_ds = xr_dataset.sel(
                            time_utc=slice(dates_array[i], dates_array[i + (self.intervals - 1)])
                        )

                        # Extracting system ids with continous NaN's
                        sys_ids = return_system_indices_which_has_contiguous_nan(
                            xr_ds.values, check_interval=self.intervals
                        )

                        # Dropping the systems which are inactive for the whole day
                        if not len(sys_ids) == 0:
                            xr_dataset = xr_dataset.drop_isel(pv_system_id=sys_ids)
            else:
                pass

            yield xr_dataset
