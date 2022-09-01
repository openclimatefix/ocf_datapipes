"""Converts Satellite to int8 for Power Perceiver"""
import numpy as np
import xarray as xr
from torchdata.datapipes import functional_datapipe
from torchdata.datapipes.iter import IterDataPipe


@functional_datapipe("convert_satellite_to_int8")
class ConvertSatelliteToInt8IterDataPipe(IterDataPipe):
    """Converts satellite to int8"""

    def __init__(self, source_datapipe: IterDataPipe):
        """
        Converts satellite to int8

        Args:
            source_datapipe: Datapipe of satellite data
        """
        super().__init__()
        self.source_datapipe = source_datapipe

    def __iter__(self) -> xr.DataArray:
        for xr_dataset in self.source_datapipe:
            xr_dataset = xr_dataset.clip(min=0, max=1023)
            xr_dataset.data = (xr_dataset.astype(np.float32).data / 4.0).round().astype(np.uint8)
            yield xr_dataset
