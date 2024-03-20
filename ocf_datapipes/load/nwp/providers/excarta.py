"""Excarta Loading"""
import pandas as pd
import xarray as xr

from ocf_datapipes.load.nwp.providers.utils import open_zarr_paths


def preprocess_excarta(ds: xr.Dataset) -> xr.Dataset:
    """
    Preprocesses the Excarta hindcast data
    Args:
        ds: The dataset to preprocess

    Returns:
        The preprocessed dataset, with the init_time_utc added
    """
    # Filename
    filename = ds.encoding["source"]
    # Get the init time from the filename
    init_time = pd.to_datetime(filename.split("/")[-1].split(".")[0])
    # Set the init time
    ds["init_time_utc"] = init_time
    return ds


def open_excarta(zarr_path) -> xr.Dataset:
    """
    Opens the Excarta hindcast data

    Args:
        zarr_path: Path to the zarr to open

    Returns:
        Xarray DataArray of the NWP data
    """
    # Open the data
    nwp = open_zarr_paths(zarr_path, time_dim="init_time_utc", preprocessor=preprocess_excarta)
    nwp = nwp.rename({"prediction_timedelta": "step"})
    nwp = nwp.sortby("init_time_utc")
    # Sanity checks.
    time = pd.DatetimeIndex(nwp.init_time_utc)
    assert time.is_unique
    assert time.is_monotonic_increasing
    return nwp
