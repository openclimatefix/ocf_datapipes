import numpy as np
import xarray as xr
from torchdata.datapipes import functional_datapipe
from torchdata.datapipes.iter import IterDataPipe


@functional_datapipe("convert_satellite_to_int8")
class ConvertSatelliteToInt8IterDataPipe(IterDataPipe):
    def __init__(self, source_dp: IterDataPipe):
        super().__init__()
        self.source_dp = source_dp

    def __iter__(self):
        for xr_dataset in self.source_dp:
            xr_dataset = xr_dataset.clip(min=0, max=1023)
            xr_dataset.data = (xr_dataset.astype(np.float32).data / 4.0).round().astype(np.uint8)
            yield xr_dataset
