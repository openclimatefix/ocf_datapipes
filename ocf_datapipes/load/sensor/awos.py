"""Load ASOS data from local files for training/inference"""

import logging

import fsspec
import xarray as xr
from torch.utils.data import IterDataPipe, functional_datapipe

from ocf_datapipes.config.model import Sensor

_log = logging.getLogger(__name__)


@functional_datapipe("OpenAWOS")
class OpenAWOSFromNetCDFIterDataPipe(IterDataPipe):
    """OpenAWOSFromNetCDFIterDataPipe"""

    def __init__(
        self,
        sensor: Sensor,
    ):
        """
        Datapipe to load sensor data from AWOS network

        Args:
            sensor: Sensor configuration
        """
        super().__init__()
        self.sensor = sensor
        self.filename = self.sensor.sensor_filename
        self.variables = list(self.sensor.sensor_variables)

    def __iter__(self):
        with fsspec.open(self.filename, "rb") as f:
            ds = xr.open_dataset(f)
            # Rename timestamp to time_utc
            ds = ds.rename({"timestamp": "time_utc"})
            # Add coordinate to station_id dimension
            ds = ds.assign_coords(station_id=list(range(len(ds.station_id))))
            # Get all indicies where the latitude is NaN
            nan_lat = ds.latitude.isnull().values
            nan_lat_indicies = nan_lat.nonzero()[0]
            ds = ds.drop_isel(station_id=nan_lat_indicies)
            # Only keep wind speed
            ds = ds[self.variables].to_array()
        while True:
            yield ds
