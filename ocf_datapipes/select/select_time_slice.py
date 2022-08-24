from datetime import timedelta
from typing import Union

import xarray as xr
from torchdata.datapipes import functional_datapipe
from torchdata.datapipes.iter import IterDataPipe, Zipper


@functional_datapipe("select_time_slice")
class SelectTimeSliceIterDataPipe(IterDataPipe):
    def __init__(
        self,
        source_datapipe: IterDataPipe,
        t0_datapipe: IterDataPipe,
        history_duration: timedelta,
        forecast_duration: timedelta,
    ):
        self.source_datapipe = source_datapipe
        self.t0_datapipe = t0_datapipe
        self.history_duration = history_duration
        self.forecast_duration = forecast_duration

    def __iter__(self) -> Union[xr.DataArray, xr.Dataset]:
        for xr_data, t0 in Zipper(self.source_datapipe, self.t0_datapipe):
            xr_data = xr_data.sel(
                time_utc=slice(t0 - self.history_duration, t0 + self.forecast_duration)
            )
            yield xr_data
