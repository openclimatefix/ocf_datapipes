"""Open GFS Forecast data"""
import logging
from pathlib import Path
from typing import Union

import xarray as xr
from torchdata.datapipes import functional_datapipe
from torchdata.datapipes.iter import IterDataPipe

_log = logging.getLogger(__name__)


@functional_datapipe("open_gfs")
class OpenGFSForecastIterDataPipe(IterDataPipe):
    """Open GFS Forecast data"""

    def __init__(self, zarr_path: Union[Path, str]):
        """
        Open GFS Forecast data

        Args:
            zarr_path: Path or wildcard path to GFS Zarrs
        """
        self.zarr_path = zarr_path

    def __iter__(self):
        _log.debug("Opening NWP data: %s", self.zarr_path)
        gfs = open_gfs(self.zarr_path)
        while True:
            yield gfs


def open_gfs(zarr_path: Union[Path, str]) -> xr.Dataset:
    """
    Opens GFS dataset

    Args:
        zarr_path: Path to Zarr(s) to open

    Returns:
        Xarray dataset of GFS Forecasts
    """
    if "*" in zarr_path:
        gfs = xr.open_mfdataset(zarr_path, engine="zarr", combine="time", chunks="auto")
    else:
        gfs = xr.open_dataset(zarr_path, engine="zarr", mode="r", chunks="auto")

    gfs = gfs.rename({"time": "time_utc"})
    # TODO Do other variables
    return gfs
