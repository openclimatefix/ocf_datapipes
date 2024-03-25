"""Common NWP providers"""

from typing import Callable

import xarray as xr


def open_zarr_paths(zarr_path, time_dim="init_time", preprocessor: Callable = None) -> xr.Dataset:
    """
    Opens the NWP data

    Args:
        zarr_path: Path to the zarr(s) to open
        time_dim: Name of the time dimension
        preprocessor: Optional preprocessor to apply to the dataset

    Returns:
        The opened Xarray Dataset
    """
    if type(zarr_path) in [list, tuple] or "*" in str(zarr_path):  # Multi-file dataset
        nwp = xr.open_mfdataset(
            zarr_path,
            engine="zarr",
            concat_dim=time_dim,
            combine="nested",
            chunks="auto",
            preprocess=preprocessor,
        ).sortby(time_dim)
    else:
        nwp = xr.open_dataset(
            zarr_path,
            engine="zarr",
            consolidated=True,
            mode="r",
            chunks="auto",
        )
    return nwp
