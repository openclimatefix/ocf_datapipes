"""ECMWF provider loaders"""
import pandas as pd
import xarray as xr

from ocf_datapipes.load.nwp.providers.utils import open_zarr_paths


def open_ifs(zarr_path) -> xr.DataArray:
    """
    Opens the ECMWF IFS NWP data

    Args:
        zarr_path: Path to the zarr to open

    Returns:
        Xarray DataArray of the NWP data
    """
    # Open the data
    nwp = open_zarr_paths(zarr_path)
    # TODO This name should change to be something not UKV
    ifs: xr.DataArray = nwp["UKV"]
    del nwp
    ifs = ifs.transpose("init_time", "step", "variable", "latitude", "longitude")
    ifs = ifs.rename(
        {
            "init_time": "init_time_utc",
            "variable": "channel",
            "latitude": "latitude",
            "longitude": "longitude",
        }
    )
    # Sanity checks.
    time = pd.DatetimeIndex(ifs.init_time_utc)
    assert time.is_unique
    assert time.is_monotonic_increasing
    return ifs
