"""

# Resample to 5-minutely and interpolate up to 15 minutes ahead.
    # TODO: Issue #74: Give users the option to NOT resample (because Perceiver IO
    # doesn't need all the data to be perfectly aligned).
    pv_power_watts = pv_power_watts.resample("5T").interpolate(method="time", limit=3)
    pv_power_watts.dropna(axis="index", how="all", inplace=True)
    pv_power_watts.dropna(axis="columns", how="all", inplace=True)

"""

from typing import Union

import xarray as xr
from torchdata.datapipes import functional_datapipe
from torchdata.datapipes.iter import IterDataPipe


@functional_datapipe("resample")
class ResampleIterDataPipe(IterDataPipe):
    def __init__(
        self, source_datapipe: IterDataPipe, data_name: str, frequency: str, method: str, limit: int
    ):
        self.source_datapipe = source_datapipe
        self.frequency = frequency
        self.method = method

    def __iter__(self) -> Union[xr.DataArray, xr.Dataset]:
        for xr_data in self.source_datapipe:
            # Drop based off capacity here

            yield xr_data
