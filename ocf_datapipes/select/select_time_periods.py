from typing import Union

import pandas as pd
import xarray as xr
from torchdata.datapipes import functional_datapipe
from torchdata.datapipes.iter import IterDataPipe


@functional_datapipe("select_time_periods")
class SelectTimePeriodsIterDataPipe(IterDataPipe):
    def __init__(
        self, source_dp: IterDataPipe, time_periods: pd.DataFrame, dim_name: str = "time_utc"
    ):
        self.source_dp = source_dp
        self.time_periods = time_periods
        self.dim_name = dim_name

    def __iter__(self) -> Union[xr.DataArray, xr.Dataset]:
        for xr_data in self.source_dp:
            new_xr_data = []
            for _, row in self.time_periods.iterrows():
                start_dt = row["start_dt"]
                end_dt = row["end_dt"]
                new_xr_data.append(xr_data.sel({self.dim_name: slice(start_dt, end_dt)}))
            yield xr.concat(new_xr_data, dim=self.dim_name)
