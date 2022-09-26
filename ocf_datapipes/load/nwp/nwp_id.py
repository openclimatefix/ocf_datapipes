"""NWP Loader"""
import logging
from pathlib import Path
from typing import Union

import pandas as pd
import xarray as xr
from torchdata.datapipes import functional_datapipe
from torchdata.datapipes.iter import IterDataPipe

_log = logging.getLogger(__name__)


@functional_datapipe("open_nwp_id")
class OpenNWPIDIterDataPipe(IterDataPipe):
    """Opens NWP Zarr and yields it"""

    def __init__(self, netcdf_path: Union[Path, str]):
        """
        Opens NWP Zarr and yields it

        Args:
            netcdf_path: Path to the netcdf file
        """
        self.netcdf_path = netcdf_path

    def __iter__(self) -> Union[xr.DataArray, xr.Dataset]:
        """Opens the NWP data"""
        _log.debug("Opening NWP data: %s", self.netcdf_path)
        ukv = open_nwp(self.netcdf_path)
        while True:
            yield ukv


def open_nwp(netcdf_path) -> xr.DataArray:
    """
    Opens the NWP data

    Args:
        zarr_path: Path to the zarr to open

    Returns:
        Xarray DataArray of the NWP data
    """
    _log.debug("Loading NWP")
    nwp = xr.load_dataset(
        netcdf_path,
        engine="h5netcdf",
        chunks="auto",
    )
    ukv: xr.DataArray = nwp["UKV"]
    del nwp
    ukv = ukv.transpose("init_time", "step", "variable", "id")
    ukv = ukv.rename({"init_time": "init_time_utc", "variable": "channel"})

    _log.debug("Resampling to 1 hour")
    ukv = ukv.resample(init_time_utc="1H").pad()

    # Sanity checks.
    time = pd.DatetimeIndex(ukv.init_time_utc)
    assert time.is_unique
    assert time.is_monotonic_increasing
    return ukv
