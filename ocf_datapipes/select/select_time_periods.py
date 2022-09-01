"""Select time periods"""
from typing import Union

import pandas as pd
import xarray as xr
from torchdata.datapipes import functional_datapipe
from torchdata.datapipes.iter import IterDataPipe


@functional_datapipe("select_time_periods")
class SelectTimePeriodsIterDataPipe(IterDataPipe):
    """Select time periods"""

    def __init__(
        self, source_datapipe: IterDataPipe, time_periods: pd.DataFrame, dim_name: str = "time_utc"
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
        for xr_data in self.source_datapipe:
            new_xr_data = []
            for _, row in self.time_periods.iterrows():
                start_dt = row["start_dt"]
                end_dt = row["end_dt"]
                new_xr_data.append(xr_data.sel({self.dim_name: slice(start_dt, end_dt)}))
            yield xr.concat(new_xr_data, dim=self.dim_name)
