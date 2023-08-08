import pandas as pd
import xarray as xr

from ocf_datapipes.load.nwp.providers.utils import open_zarr_paths


def open_icon_eu(zarr_path) -> xr.Dataset:
    """
    Opens the ICON data

    ICON EU Data is on a regular lat/lon grid
    It has data on multiple pressure levels, as well as the surface
    Each of the variables is its own data variable

    Args:
        zarr_path: Path to the zarr to open

    Returns:
        Xarray DataArray of the NWP data
    """
    # Open the data
    nwp = open_zarr_paths(zarr_path, time_dim="time")
    nwp = nwp.rename({"time": "init_time_utc"})
    # Sanity checks.
    time = pd.DatetimeIndex(nwp.init_time_utc)
    assert time.is_unique
    assert time.is_monotonic_increasing
    return nwp


def open_icon_global(zarr_path) -> xr.DataArray:
    """
    Opens the ICON data

    ICON Global Data is on an isohedral grid, so the points are not regularly spaced
    It has data on multiple pressure levels, as well as the surface
    Each of the variables is its own data variable

    Args:
        zarr_path: Path to the zarr to open

    Returns:
        Xarray DataArray of the NWP data
    """
    # Open the data
    open_zarr_paths(zarr_path)
    raise NotImplementedError("ICON data is not yet supported")
