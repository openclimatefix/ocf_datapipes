"""Load ASOS data from local files for training/inference"""
import logging

import fsspec
import xarray as xr
from torch.utils.data import IterDataPipe, functional_datapipe

from ocf_datapipes.config.model import Sensor

_log = logging.getLogger(__name__)


@functional_datapipe("OpenMeteomatics")
class OpenMeteomaticsFromNetCDFIterDataPipe(IterDataPipe):
    """OpenMeteomaticsFromNetCDFIterDataPipe"""

    def __init__(
        self,
        sensor: Sensor,
    ):
        """
        Datapipe to load Meteomatics point data

        Args:
            sensor: Sensor configuration
        """
        super().__init__()
        self.sensor = sensor
        self.filename = self.sensor.sensor_filename
        self.variables = list(self.sensor.sensor_variables)

    def __iter__(self):
        with fsspec.open(self.filename, "rb") as f:
            ds = xr.open_mfdataset(f, engine="zarr", combine="nested", concat_dim="time_utc")
            ds = ds.sortby("time_utc")
            ds = ds[self.variables].to_array()
        while True:
            yield ds
