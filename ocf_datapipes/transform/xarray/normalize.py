import xarray as xr
from torchdata.datapipes import functional_datapipe
from torchdata.datapipes.iter import IterDataPipe


@functional_datapipe("normalize")
class NormalizeIterDataPipe(IterDataPipe):
    def __init__(self, source_dp: IterDataPipe, mean = None, std = None, max_value = None):
        self.source_dp = source_dp
        self.mean = mean
        self.std = std
        self.max_value = max_value

    def __iter__(self):
        for xr_data in self.source_dp:
            if self.mean is not None and self.std is not None:
                xr_data = xr_data - self.mean
                xr_data = xr_data / self.std
            elif self.max_value is not None:
                xr_data = xr_data / self.max_value
            yield xr_data
