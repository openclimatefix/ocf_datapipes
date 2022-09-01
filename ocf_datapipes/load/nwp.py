"""NWP Loader"""
import logging
from pathlib import Path
from typing import Union

import numpy as np
import pandas as pd
import xarray as xr
from torchdata.datapipes import functional_datapipe
from torchdata.datapipes.iter import IterDataPipe

_log = logging.getLogger(__name__)


@functional_datapipe("open_nwp")
class OpenNWPIterDataPipe(IterDataPipe):
    """Opens NWP Zarr and yields it"""

    def __init__(self, zarr_path: Union[Path, str]):
        """
        Opens NWP Zarr and yields it

        Args:
            zarr_path: Path to the Zarr file
        """
        self.zarr_path = zarr_path

    def __iter__(self) -> Union[xr.DataArray, xr.Dataset]:
        """Opens the NWP data"""
        _log.debug("Opening NWP data: %s", self.zarr_path)
        ukv = open_nwp(self.zarr_path)
        while True:
            yield ukv


def open_nwp(zarr_path) -> xr.DataArray:
    """
    Opens the NWP data

    Args:
        zarr_path: Path to the zarr to open

    Returns:
        Xarray DataArray of the NWP data
    """
    nwp = xr.open_dataset(
        zarr_path,
        engine="zarr",
        consolidated=True,
        mode="r",
        chunks="auto",
    )
    ukv: xr.DataArray = nwp["UKV"]
    del nwp
    ukv = ukv.transpose("init_time", "step", "variable", "y", "x")
    ukv = ukv.rename(
        {"init_time": "init_time_utc", "variable": "channel", "y": "y_osgb", "x": "x_osgb"}
    )
    # y_osgb and x_osgb are int64 on disk.
    for coord_name in ("y_osgb", "x_osgb"):
        ukv[coord_name] = ukv[coord_name].astype(np.float32)
    # Sanity checks.
    assert ukv.y_osgb[0] > ukv.y_osgb[1], "UKV must run from top-to-bottom."
    time = pd.DatetimeIndex(ukv.init_time_utc)
    assert time.is_unique
    assert time.is_monotonic_increasing
    return ukv
