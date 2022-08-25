from typing import Union

import pandas as pd
import xarray as xr
from torchdata.datapipes import functional_datapipe
from torchdata.datapipes.iter import IterDataPipe, Zipper


@functional_datapipe("add_nwp_target_time")
class AddNWPTargetTimeIterDataPipe(IterDataPipe):
    def __init__(
        self,
        source_datapipe: IterDataPipe,
        t0_datapipe: IterDataPipe,
        sample_period_duration,
        history_duration,
        forecast_duration,
    ):
        self.source_datapipe = source_datapipe
        self.t0_datapipe = t0_datapipe
        self.sample_period_duration = sample_period_duration
        self.history_duration = history_duration
        self.forecast_duration = forecast_duration

    def __iter__(self) -> Union[xr.DataArray, xr.Dataset]:
        for xr_data, t0 in Zipper(self.source_datapipe, self.t0_datapipe):
            t0_datetime_utc = pd.Timestamp(t0)
            start_dt = t0_datetime_utc - self.history_duration
            end_dt = t0_datetime_utc + self.forecast_duration
            target_times = pd.date_range(
                start_dt.ceil(self.sample_period_duration),
                end_dt.ceil(self.sample_period_duration),
                freq=self.sample_period_duration,
            )
            xr_data.attrs["target_time_utc"] = target_times
            yield xr_data
