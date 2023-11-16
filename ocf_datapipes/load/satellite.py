"""Satellite loader"""
import logging
import subprocess
from datetime import datetime, timedelta
from pathlib import Path
from typing import Union

import dask
import fsspec
import pandas as pd
import xarray as xr
from ocf_blosc2 import Blosc2  # noqa: F401
from pathy import Pathy
from torch.utils.data import IterDataPipe, functional_datapipe

_log = logging.getLogger(__name__)


def _get_single_sat_data(zarr_path: Union[Path, str]) -> xr.DataArray:
    """Helper function to open a zarr from either local or GCP path.

    The local or GCP path may contain wildcard matching (*)

    Args:
        zarr_path: Path to zarr file
    """

    # These kwargs are used if zarr path contains "*"
    openmf_kwargs = dict(
        engine="zarr",
        concat_dim="time",
        combine="nested",
        chunks="auto",
        join="override",
    )

    # Need to generate list of files if using GCP bucket storage
    if "gs://" in str(zarr_path) and "*" in str(zarr_path):
        result_string = subprocess.run(
            f"gsutil ls -d {zarr_path}".split(" "), stdout=subprocess.PIPE
        ).stdout.decode("utf-8")
        files = result_string.splitlines()

        dataset = xr.open_mfdataset(files, **openmf_kwargs)

    elif "*" in str(zarr_path):  # Multi-file dataset
        dataset = xr.open_mfdataset(zarr_path, **openmf_kwargs)
    else:
        dataset = xr.open_dataset(zarr_path, engine="zarr", chunks="auto")
    dataset = dataset.drop_duplicates("time").sortby("time")

    return dataset


def open_sat_data(
    zarr_path: Union[Path, str, list[Path], list[str]], use_15_minute_data_if_needed: bool = False
) -> xr.DataArray:
    """Lazily opens the Zarr store.

    Args:
      zarr_path: Cloud URL or local path pattern, or list of these. If GCS URL, it must start with
          'gs://'.
      use_15_minute_data_if_needed: use_15_minute_data_if_needed: Option to use the 15 minute data
        if the 5 minute data is not available
        This is done by checking to see if the last timestamp is within an hour from now

    Example:
        With wild cards and GCS path:
        ```
        zarr_paths = [
            "gs://bucket/2020_nonhrv_split_*.zarr",
            "gs://bucket/2019_nonhrv_split_*.zarr",
        ]
        ds = open_sat_data(zarr_paths)
        ```
        Without wild cards and with local path:
        ```
        zarr_paths = [
            "/data/2020_nonhrv.zarr",
            "/data/2019_nonhrv.zarr",
        ]
        ds = open_sat_data(zarr_paths)
        ```
    """
    _log.info(f"Opening satellite data: %s, {use_15_minute_data_if_needed=}", zarr_path)

    # Silence the warning about large chunks.
    # Alternatively, we could set this to True, but that slows down loading a Satellite batch
    # from 8 seconds to 50 seconds!
    dask.config.set({"array.slicing.split_large_chunks": False})

    # dont load 15 minute data by default
    use_15_minute_data = False

    if isinstance(zarr_path, (list, tuple)):
        dataset = xr.combine_nested(
            [_get_single_sat_data(path) for path in zarr_path],
            concat_dim="time",
            combine_attrs="override",
            join="override",
        )
    else:
        # check the file exists
        dataset, use_15_minute_data = load_and_check_satellite_data(zarr_path)

    if use_15_minute_data_if_needed and use_15_minute_data:
        zarr_path_15_minutes = str(zarr_path).replace(".zarr", "_15.zarr")

        _log.info(f"Now going to load {zarr_path_15_minutes} and resample")
        dataset = _get_single_sat_data(zarr_path_15_minutes)

        dataset = dataset.load()
        _log.debug("Resampling 15 minute data to 5 mins")
        dataset = dataset.resample(time="5T").interpolate("linear")
    else:
        _log.debug("Not using 15 minute data")

    # Remove data coordinate dimensions if they exist
    if "x_geostationary_coordinates" in dataset:
        del dataset["x_geostationary_coordinates"]
        del dataset["y_geostationary_coordinates"]

    # Rename
    # These renamings will no longer be necessary when the Zarr uses the 'correct' names,
    # see https://github.com/openclimatefix/Satip/issues/66
    if "variable" in dataset:
        dataset = dataset.rename({"variable": "channel"})
    if "channels" in dataset:
        dataset = dataset.rename({"channels": "channel"})
    elif "channel" not in dataset:
        # This is HRV version 3, which doesn't have a channels dim.  So add one.
        dataset = dataset.expand_dims(dim={"channel": ["HRV"]}, axis=1)

    # Rename coords to be more explicit about exactly what some coordinates hold:
    # Note that `rename` renames *both* the coordinates and dimensions, and keeps
    # the connection between the dims and coordinates, so we don't have to manually
    # use `data_array.set_index()`.
    dataset = dataset.rename(
        {
            "time": "time_utc",
        }
    )
    if "y" in dataset.coords.keys():
        dataset = dataset.rename(
            {
                "y": "y_geostationary",
            }
        )

    if "x" in dataset.coords.keys():
        dataset = dataset.rename(
            {
                "x": "x_geostationary",
            }
        )

    # Flip coordinates to top-left first
    if dataset.y_geostationary[0] < dataset.y_geostationary[-1]:
        dataset = dataset.reindex(y_geostationary=dataset.y_geostationary[::-1])
    if dataset.x_geostationary[0] > dataset.x_geostationary[-1]:
        dataset = dataset.reindex(x_geostationary=dataset.x_geostationary[::-1])

    data_array = dataset["data"]
    del dataset

    # Ensure the y and x coords are in the right order (top-left first):
    assert data_array.y_geostationary[0] > data_array.y_geostationary[-1]
    assert data_array.x_geostationary[0] < data_array.x_geostationary[-1]
    if "y_osgb" in data_array.dims:
        assert data_array.y_osgb[0, 0] > data_array.y_osgb[-1, 0]
        assert data_array.x_osgb[0, 0] < data_array.x_osgb[0, -1]

    # Sanity checks!
    data_array = data_array.transpose("time_utc", "channel", "y_geostationary", "x_geostationary")
    assert data_array.dims == ("time_utc", "channel", "y_geostationary", "x_geostationary")
    datetime_index = pd.DatetimeIndex(data_array.time_utc)
    assert datetime_index.is_unique
    assert datetime_index.is_monotonic_increasing
    # Satellite datetimes can sometimes be 04, 09, minutes past the hour, or other slight offsets.
    # These slight offsets will break downstream code, which expects satellite data to be at
    # exactly 5 minutes past the hour.
    assert (datetime_index == datetime_index.round("5T")).all()

    _log.info("Opened satellite data")

    return data_array


def load_and_check_satellite_data(zarr_path) -> [xr.Dataset, bool]:
    """
    Load the satellite data,

    1. check if file path exists
    2. check dataset has been updated in the last hour
    If 1. or 2. are true, then return True for use_15_minute_data

    Args:
        zarr_path: the zarr path to load

    Returns:
        dataset (if loaded),
        use_15_minute_data, indicating if the 15 minute data should be loaded
    """
    filesystem = fsspec.open(Pathy.fluid(zarr_path)).fs
    if filesystem.exists(zarr_path):
        dataset = _get_single_sat_data(zarr_path)

        use_15_minute_data = check_last_timestamp(dataset)

    else:
        _log.info(f"File does not exist {zarr_path}. Will try to load 15 minute data")
        use_15_minute_data = True
        dataset = None
    return dataset, use_15_minute_data


def check_last_timestamp(dataset: xr.Dataset, timedelta_hours: float = 1) -> bool:
    """
    Check the last timestamp of the dataset to see if it is more than 1 hour ago

    Args:
        dataset: dataset with time dimension
        timedelta_hours: the timedelta to check from now

    Returns: bool
    """
    latest_time = pd.to_datetime(dataset.time[-1].values)
    now = datetime.utcnow()
    if latest_time < now - timedelta(hours=timedelta_hours):
        _log.info(
            f"last datestamp is {latest_time}, which is more than "
            f"{timedelta_hours} hour ago from {now} "
            f"Will try to load 15 minute data"
        )
        return True
    else:
        _log.debug(
            f"last datestamp is {latest_time}, which is less than {timedelta_hours} "
            f"hour ago from {now}"
        )
        return False


@functional_datapipe("open_satellite")
class OpenSatelliteIterDataPipe(IterDataPipe):
    """Open Satellite Zarr"""

    def __init__(self, zarr_path: Union[Path, str], use_15_minute_data_if_needed: bool = False):
        """
        Opens the satellite Zarr

        Args:
            zarr_path: path to the zarr file
            use_15_minute_data_if_needed: Option to use the 15 minute data if the
                5 minute data is not available
        """
        self.zarr_path = zarr_path
        self.use_15_minute_data_if_needed = use_15_minute_data_if_needed
        super().__init__()

    def __iter__(self) -> xr.DataArray:
        """Open the Zarr file"""
        data: xr.DataArray = open_sat_data(
            zarr_path=self.zarr_path, use_15_minute_data_if_needed=self.use_15_minute_data_if_needed
        )
        while True:
            yield data
