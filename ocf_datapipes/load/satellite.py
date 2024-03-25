"""Satellite loader"""

import logging
import subprocess
from pathlib import Path
from typing import Union

import dask
import pandas as pd
import xarray as xr
from ocf_blosc2 import Blosc2  # noqa: F401
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


def open_sat_data(zarr_path: Union[Path, str, list[Path], list[str]]) -> xr.DataArray:
    """Lazily opens the Zarr store.

    Args:
      zarr_path: Cloud URL or local path pattern, or list of these. If GCS URL, it must start with
          'gs://'.

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

    # Silence the warning about large chunks.
    # Alternatively, we could set this to True, but that slows down loading a Satellite batch
    # from 8 seconds to 50 seconds!
    dask.config.set({"array.slicing.split_large_chunks": False})

    if isinstance(zarr_path, (list, tuple)):
        message_files_list = "\n - " + "\n - ".join([str(s) for s in zarr_path])
        _log.info(f"Opening satellite data: {message_files_list}")
        dataset = xr.combine_nested(
            [_get_single_sat_data(path) for path in zarr_path],
            concat_dim="time",
            combine_attrs="override",
            join="override",
        )
    else:
        _log.info(f"Opening satellite data: {zarr_path}")
        dataset = _get_single_sat_data(zarr_path)

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
    assert (datetime_index == datetime_index.round("5min")).all()

    _log.info("Opened satellite data")

    return data_array


@functional_datapipe("open_satellite")
class OpenSatelliteIterDataPipe(IterDataPipe):
    """Open Satellite Zarr"""

    def __init__(self, zarr_path: Union[Path, str]):
        """
        Opens the satellite Zarr

        Args:
            zarr_path: path to the zarr file
        """
        self.zarr_path = zarr_path
        super().__init__()

    def __iter__(self) -> xr.DataArray:
        """Open the Zarr file"""
        data: xr.DataArray = open_sat_data(zarr_path=self.zarr_path)
        while True:
            yield data
