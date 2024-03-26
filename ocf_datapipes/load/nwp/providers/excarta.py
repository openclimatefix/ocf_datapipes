"""Excarta Loading"""
import numpy as np
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
    # wind is split into speed and direction, so would want to decompose it with sin and cos
    # And split into u and v
    nwp["10u"] = nwp["10m_wind_speed"] * np.cos(np.deg2rad(nwp["10m_wind_speed_angle"]))
    nwp["10v"] = nwp["10m_wind_speed"] * np.sin(np.deg2rad(nwp["10m_wind_speed_angle"]))
    nwp["100u"] = nwp["100m_wind_speed"] * np.cos(np.deg2rad(nwp["100m_wind_speed_angle"]))
    nwp["100v"] = nwp["100m_wind_speed"] * np.sin(np.deg2rad(nwp["100m_wind_speed_angle"]))
    # Sanity checks.
    time = pd.DatetimeIndex(nwp.init_time_utc)
    assert time.is_unique
    assert time.is_monotonic_increasing
    return nwp
