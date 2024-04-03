"""ECMWF provider loaders"""
import pandas as pd
import xarray as xr
import numpy as np

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
    nwp = nwp.reindex(latitude=np.concatenate([np.arange(62, 60, -0.05), nwp.latitude.values])) # hack to fill in data for shetland isles
    dataVars = list(nwp.data_vars.keys())
    if len(dataVars) > 1:
        raise Exception("Too many TLDVs")
    else:
        dataVar = dataVars[0]
    ifs: xr.Dataset = nwp[dataVar]
    del nwp
    ifs = ifs.transpose("init_time", "step", "variable", "latitude", "longitude")
    ifs = ifs.rename(
        {
            "init_time": "init_time_utc",
            "variable": "channel",
        }
    )
    # Sanity checks.
    time = pd.DatetimeIndex(ifs.init_time_utc)
    assert time.is_unique
    assert time.is_monotonic_increasing
    return ifs
