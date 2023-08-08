import xarray as xr


def open_zarr_paths(zarr_path, time_dim="init_time") -> xr.Dataset:
    if type(zarr_path) in [list, tuple] or "*" in str(zarr_path):  # Multi-file dataset
        nwp = xr.open_mfdataset(
            zarr_path,
            engine="zarr",
            concat_dim=time_dim,
            combine="nested",
            chunks={},
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
