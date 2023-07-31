import numpy as np
import pandas as pd
import xarray as xr


def open_icon(zarr_path) -> xr.DataArray:
    """
    Opens the ICON data

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
    raise NotImplementedError("ICON data is not yet supported")
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
