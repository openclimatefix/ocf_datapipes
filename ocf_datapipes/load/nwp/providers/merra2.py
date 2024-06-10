"""MERRA2 provider loaders"""

import pandas as pd
import xarray as xr

from ocf_datapipes.load.nwp.providers.utils import open_zarr_paths


def open_merra2(zarr_path) -> xr.DataArray:
    """
    Opens the MERRA2 AOD data

    Args:
        zarr_path: Path to the zarr to open

    Returns:
        Xarray DataArray of the NWP data
    """
    # Open the data
    nwp = open_zarr_paths(zarr_path)

    init_time = nwp.time[0]
    nwp = nwp.expand_dims({"init_time_utc": [init_time.values]})
    nwp = nwp.rename({"lat": "latitude", "lon": "longitude", "time": "step"})
    nwp["step"] = nwp["step"] - init_time.values
    nwp = nwp.expand_dims({"channel": list(nwp.keys())}).assign_coords(
        {"channel": list(nwp.keys())}
    )
    nwp = nwp.transpose("init_time_utc", "step", "channel", "latitude", "longitude")
    aodana: xr.DataArray = nwp["AODANA"]
    del nwp

    # Sanity checks.
    time = pd.DatetimeIndex(aodana.step + aodana.init_time_utc.values)
    assert time.is_unique
    assert time.is_monotonic_increasing
    return aodana
