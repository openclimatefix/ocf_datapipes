"""

pv_power_watts = pv_power_watts.clip(lower=0, upper=5e7)

"""

from typing import Union

import numpy as np
import xarray as xr
from torchdata.datapipes import functional_datapipe
from torchdata.datapipes.iter import IterDataPipe


@functional_datapipe("clip")
class ClipIterDataPipe(IterDataPipe):
    def __init__(
        self,
        source_datapipe: IterDataPipe,
        data_name: str,
        min: Union[int, float] = 0.0,
        max: Union[int, float] = np.inf,
    ):
        self.source_datapipe = source_datapipe
        self.min = min
        self.max = max
        self.data_name = data_name

    def __iter__(self) -> Union[xr.DataArray, xr.Dataset]:
        for xr_data in self.source_datapipe:
            # Drop based off capacity here

            yield xr_data
