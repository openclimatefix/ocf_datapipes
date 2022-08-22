import logging
from pathlib import Path
from typing import Union

import dask
import numpy as np
import pandas as pd
import xarray as xr
from torchdata.datapipes import functional_datapipe
from torchdata.datapipes.iter import IterDataPipe

_log = logging.getLogger(__name__)


def open_sat_data(
    zarr_path: Union[Path, str],
) -> xr.DataArray:
    """Lazily opens the Zarr store.

    Args:
      zarr_path: Cloud URL or local path pattern.  If GCP URL, must start with 'gs://'
    """
    _log.debug("Opening satellite data: %s", zarr_path)

    # Silence the warning about large chunks.
    # Alternatively, we could set this to True, but that slows down loading a Satellite batch
    # from 8 seconds to 50 seconds!
    dask.config.set({"array.slicing.split_large_chunks": False})

    # Open the data
    dataset = xr.open_dataset(zarr_path, engine="zarr", chunks="auto")

    # Flip coordinates to top-left first
    dataset = dataset.reindex(y=dataset.y[::-1])
    dataset = dataset.reindex(x=dataset.x[::-1])

    # Rename
    # These renamings will no longer be necessary when the Zarr uses the 'correct' names,
    # see https://github.com/openclimatefix/Satip/issues/66
    if "variable" in dataset:
        dataset = dataset.rename({"variable": "channel"})
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
            "y": "y_geostationary",
            "x": "x_geostationary",
        }
    )

    data_array = dataset["data"]
    del dataset

    # Ensure the y and x coords are in the right order (top-left first):
    assert data_array.y_geostationary[0] > data_array.y_geostationary[-1]
    assert data_array.x_geostationary[0] < data_array.x_geostationary[-1]
    assert data_array.y_osgb[0, 0] > data_array.y_osgb[-1, 0]
    assert data_array.x_osgb[0, 0] < data_array.x_osgb[0, -1]

    # Sanity checks!
    assert data_array.dims == ("time_utc", "channel", "y_geostationary", "x_geostationary")
    datetime_index = pd.DatetimeIndex(data_array.time_utc)
    assert datetime_index.is_unique
    assert datetime_index.is_monotonic_increasing
    # Satellite datetimes can sometimes be 04, 09, minutes past the hour, or other slight offsets.
    # These slight offsets will break downstream code, which expects satellite data to be at
    # exactly 5 minutes past the hour.
    assert (datetime_index == datetime_index.round("5T")).all()

    return data_array


@functional_datapipe("open_satellite")
class OpenSatelliteDataPipe(IterDataPipe):
    def __int__(self, zarr_path: Union[Path, str]) -> None:
        super().__int__()
        self.zarr_path = zarr_path

    def __iter__(self) -> xr.DataArray:
        data: xr.DataArray = open_sat_data(zarr_path=self.zarr_path)
        while True:
            yield data
