"""Select time periods"""

import datetime
import logging
from typing import Union

import xarray as xr
from torch.utils.data import IterDataPipe, functional_datapipe

logger = logging.getLogger(__name__)


@functional_datapipe("filter_times")
class FilterTimesIterDataPipe(IterDataPipe):
    """Select time periods"""

    def __init__(
        self,
        source_datapipe: IterDataPipe,
        start_time: datetime.datetime,
        end_time: datetime.datetime,
        dim_name: str = "time_utc",
    ):
        """
        Select time periods

        Args:
            source_datapipe: Datapipe of Xarray objects
            start_time: Start time to select
            end_time: End time to select
            dim_name: Dimension name for time
        """
        self.source_datapipe = source_datapipe
        self.start_time = start_time
        self.end_time = end_time
        self.dim_name = dim_name

    def __iter__(self) -> Union[xr.DataArray, xr.Dataset]:
        for xr_data in self.source_datapipe:
            logger.debug(f"Selecting Train/Test Time period ({self.start_time} - {self.end_time})")
            xr_data = xr_data.sel({self.dim_name: slice(self.start_time, self.end_time)})
            yield xr_data
