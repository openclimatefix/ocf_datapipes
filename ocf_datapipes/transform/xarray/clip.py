"""

pv_power_watts = pv_power_watts.clip(lower=0, upper=5e7)

"""

from torchdata.datapipes.iter import IterDataPipe
from torchdata.datapipes import functional_datapipe
import xarray as xr
import numpy as np
from typing import Union

@functional_datapipe("clip")
class ClipIterDataPipe(IterDataPipe):
    def __init__(self, source_datapipe: IterDataPipe, data_name: str, min: Union[int, float] = 0.0, max: Union[int, float] = np.inf):
        self.source_datapipe = source_datapipe
        self.min = min
        self.max = max
        self.data_name = data_name

    def __iter__(self) -> Union[xr.DataArray, xr.Dataset]:
        for xr_data in self.source_datapipe:
            # Drop based off capacity here

            yield xr_data
