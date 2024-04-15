"""Load Metoematics data from local files for training/inference"""
import logging

import fsspec
import xarray as xr
from torch.utils.data import IterDataPipe, functional_datapipe
import numpy as np
import ocf_blosc2

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
            data = xr.open_mfdataset(f, engine="zarr", combine="nested", concat_dim="time_utc")
            data = data.sortby("time_utc")
            # Generate U and V components from wind speed and direction
            data["100u"] = data["wind_speed_100m:ms"] * np.cos(np.deg2rad(data["wind_dir_100m:d"]))
            data["100v"] = data["wind_speed_100m:ms"] * np.sin(np.deg2rad(data["wind_dir_100m:d"]))
            data["10u"] = data["wind_speed_10m:ms"] * np.cos(np.deg2rad(data["wind_dir_10m:d"]))
            data["10v"] = data["wind_speed_10m:ms"] * np.sin(np.deg2rad(data["wind_dir_10m:d"]))
            data["200u"] = data["wind_speed_200m:ms"] * np.cos(np.deg2rad(data["wind_dir_200m:d"]))
            data["200v"] = data["wind_speed_200m:ms"] * np.sin(np.deg2rad(data["wind_dir_200m:d"]))
            data = data[self.variables].to_array()
        while True:
            yield data

