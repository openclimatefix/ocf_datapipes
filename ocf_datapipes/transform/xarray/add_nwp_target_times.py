from torchdata.datapipes.iter import IterDataPipe
from torchdata.datapipes import functional_datapipe
import xarray as xr
from typing import Union
import pandas as pd


@functional_datapipe("add_t0_idx_and_sample_period_duration")
class AddNWPTargetTimeIterDataPipe(IterDataPipe):
    def __init__(self, source_datapipe: IterDataPipe, sample_period_duration, history_duration, forecast_duration):
        self.source_datapipe = source_datapipe
        self.sample_period_duration = sample_period_duration
        self.history_duration = history_duration
        self.forecast_duration = forecast_duration

    def __iter__(self) -> Union[xr.DataArray, xr.Dataset]:
        for xr_data in self.source_datapipe:
            #target_times = pd.date_range(start_dt_ceil, end_dt_ceil, freq=self.sample_period_duration)
            yield xr_data
