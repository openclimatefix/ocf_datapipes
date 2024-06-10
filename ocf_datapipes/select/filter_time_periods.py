"""Select time periods"""

import logging
from typing import Union

import xarray as xr
from torch.utils.data import IterDataPipe, functional_datapipe

logger = logging.getLogger(__name__)


@functional_datapipe("filter_time_periods")
class FilterTimePeriodsIterDataPipe(IterDataPipe):
    """Select time periods"""

    def __init__(
        self,
        source_datapipe: IterDataPipe,
        time_periods: IterDataPipe,
        dim_name: str = "time_utc",
    ):
        """
        Select time periods

        Args:
            source_datapipe: Datapipe of Xarray objects
            time_periods: Time periods to select
            dim_name: Dimension name for time
        """
        self.source_datapipe = source_datapipe
        self.time_periods = time_periods
        self.dim_name = dim_name

    def __iter__(self) -> Union[xr.DataArray, xr.Dataset]:
        for xr_data, time_periods in self.source_datapipe.zip_ocf(self.time_periods):
            new_xr_data = []
            logger.debug(f"Selecting Time periods ({len(time_periods)})")
            for _, row in time_periods.iterrows():
                start_dt = row["start_dt"]
                end_dt = row["end_dt"]
                new_xr_data.append(xr_data.sel({self.dim_name: slice(start_dt, end_dt)}))

            xr_data_concat = xr.concat(new_xr_data, dim=self.dim_name)
            yield xr_data_concat
