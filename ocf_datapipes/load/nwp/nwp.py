"""NWP Loader"""
import logging
from pathlib import Path
from typing import Union

import numpy as np
import pandas as pd
import xarray as xr
from ocf_blosc2 import Blosc2  # noqa: F401
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


@functional_datapipe("open_latest_nwp")
class OpenLatestNWPDataPipe(IterDataPipe):
    """Yields the most recent observation from NWP data"""

    def __init__(self, base_nwp_datapipe: OpenNWPIterDataPipe) -> None:
        """Selects most recent observation from NWP data

        Args:
            base_nwp_datapipe (OpenNWPIterDataPipe): Base DataPipe, opening zarr
        """
        self.base_nwp_datapipe = base_nwp_datapipe

    def __iter__(self) -> Union[xr.DataArray, xr.Dataset]:
        """Selects most recent entry

        Returns:
            Union[xr.DataArray, xr.Dataset]: NWP slice

        Yields:
            Iterator[Union[xr.DataArray, xr.Dataset]]: Iterator of most recent NWP data
        """
        for nwp_data in self.base_nwp_datapipe:
            _nwp = nwp_data.sel(init_time_utc=nwp_data.init_time_utc.max())
            time = _nwp.init_time_utc.values
            _log.debug(f"Selected most recent NWP observation, at: {time}")
            yield _nwp


def open_nwp(zarr_path) -> xr.DataArray:
    """
    Opens the NWP data

    Args:
        zarr_path: Path to the zarr to open

    Returns:
        Xarray DataArray of the NWP data
    """
    # Open the data
    if type(zarr_path) in [list, tuple] or "*" in str(zarr_path):  # Multi-file dataset
        nwp = xr.open_mfdataset(
            zarr_path,
            engine="zarr",
            concat_dim="init_time",
            combine="nested",
            chunks={},
        ).sortby("init_time")
    else:
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
