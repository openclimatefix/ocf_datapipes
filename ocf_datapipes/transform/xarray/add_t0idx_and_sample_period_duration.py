from torchdata.datapipes.iter import IterDataPipe
from torchdata.datapipes import functional_datapipe
import xarray as xr
from typing import Union


@functional_datapipe("add_t0_idx_and_sample_period_duration")
class AddT0IdxAndSamplePeriodDurationIterDataPipe(IterDataPipe):
    def __init__(self, source_datapipe: IterDataPipe, sample_period_duration, history_duration):
        self.source_datapipe = source_datapipe
        self.sample_period_duration = sample_period_duration
        self.history_duration = history_duration
        self.t0_idx = int(self.history_duration / self.sample_period_duration)

    def __iter__(self) -> Union[xr.DataArray, xr.Dataset]:
        for xr_data in self.source_datapipe:
            xr_data.attrs["t0_idx"] = self.t0_idx
            xr_data.attrs["sample_period_duration"] = self.sample_period_duration
            yield xr_data
