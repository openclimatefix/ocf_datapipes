"""UKV provider loaders"""

import numpy as np
import pandas as pd
import xarray as xr

from ocf_datapipes.load.nwp.providers.utils import open_zarr_paths


def open_ukv(zarr_path) -> xr.DataArray:
    """
    Opens the NWP data

    Args:
        zarr_path: Path to the zarr to open

    Returns:
        Xarray DataArray of the NWP data
    """
    # Open the data
    nwp = open_zarr_paths(zarr_path)
    ukv: xr.DataArray = nwp["UKV"]
    del nwp
    ukv = ukv.transpose("init_time", "step", "variable", "y", "x")
    ukv = ukv.rename(
        {
            "init_time": "init_time_utc",
            "variable": "channel",
            "y": "y_osgb",
            "x": "x_osgb",
        }
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
