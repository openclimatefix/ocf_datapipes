from torchdata.datapipes.iter import IterDataPipe
from torchdata.datapipes import functional_datapipe
import xarray as xr

@functional_datapipe("normalize")
class NormalizeIterDataPipe(IterDataPipe):
    def __init__(self, source_dp: IterDataPipe, mean, std):
        self.source_dp = source_dp
        self.mean = mean
        self.std = std

    def __iter__(self):
        for xr_data in self.source_dp:
            xr_data = xr_data - self.mean
            xr_data = xr_data / self.std
            yield xr_data
