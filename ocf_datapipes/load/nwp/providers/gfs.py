"""Open GFS Forecast data"""

import logging
from pathlib import Path
from typing import Union

import pandas as pd
import xarray as xr

_log = logging.getLogger(__name__)


def open_gfs(zarr_path: Union[Path, str]) -> xr.Dataset:
    """
    Opens GFS dataset

    Args:
        zarr_path: Path to Zarr(s) to open

    Returns:
        Xarray dataset of GFS Forecasts
    """

    _log.info("Loading NWP GFS data")

    if "*" in zarr_path:
        nwp = xr.open_mfdataset(zarr_path, engine="zarr", combine="time", chunks="auto")
    else:
        nwp = xr.load_dataset(zarr_path, engine="zarr", mode="r", chunks="auto")

    variables = list(nwp.keys())

    nwp = xr.concat([nwp[v] for v in variables], "channel")
    nwp = nwp.assign_coords(channel=variables)

    nwp = nwp.transpose("time", "step", "channel", "latitude", "longitude")
    nwp = nwp.rename({"time": "init_time_utc"})
    nwp = nwp.transpose("init_time_utc", "step", "channel", "latitude", "longitude")
    if "valid_time" in nwp.coords.keys():
        nwp = nwp.drop("valid_time")

    _log.debug("Interpolating hour 0 to NWP data")
    nwp_step0 = nwp.interp(step=[pd.Timedelta(hours=0)])
    nwp = xr.concat([nwp_step0, nwp], dim="step")
    nwp = nwp.resample(init_time_utc="60min").pad()
    nwp = nwp.resample(step="60min").pad()

    _log.debug(nwp)

    return nwp
