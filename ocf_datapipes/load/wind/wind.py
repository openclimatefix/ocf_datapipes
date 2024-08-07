"""Datapipe and utils to load PV data from NetCDF for training"""

import io
import logging
from pathlib import Path
from typing import Union

import fsspec
import numpy as np
import pandas as pd
import xarray as xr
from torch.utils.data import IterDataPipe, functional_datapipe

from ocf_datapipes.config.model import Wind
from ocf_datapipes.load.wind.utils import (
    put_wind_data_into_an_xr_dataarray,
)

_log = logging.getLogger(__name__)


@functional_datapipe("open_wind_netcdf")
class OpenWindFromNetCDFIterDataPipe(IterDataPipe):
    """Datapipe to load NetCDF"""

    def __init__(
        self,
        wind: Wind,
    ):
        """Datapipe to load Wind from NetCDF.

        Args:
            wind: wind configuration
        """
        super().__init__()
        self.wind = wind
        self.wind_power_filenames = [
            wind_files_group.wind_filename for wind_files_group in wind.wind_files_groups
        ]
        self.wind_metadata_filenames = [
            wind_files_group.wind_metadata_filename for wind_files_group in wind.wind_files_groups
        ]

    def __iter__(self):
        wind_array_list = []
        for i in range(len(self.wind_power_filenames)):
            wind_array: xr.DataArray = load_everything_into_ram(
                self.wind_power_filenames[i],
                self.wind_metadata_filenames[i],
            )
            wind_array_list.append(wind_array)

        wind_array = xr.concat(wind_array_list, dim="wind_system_id")

        while True:
            yield wind_array


def _load_wind_metadata(filename: str) -> pd.DataFrame:
    """Return pd.DataFrame of PV metadata.

    Shape of the returned pd.DataFrame for Passiv PV data:
        Index: system_id
        Columns: latitude, longitude, ml_id
    """
    _log.info(f"Loading Wind metadata from {filename}")

    index_col = "system_id"
    df_metadata = pd.read_csv(filename, index_col=index_col)

    # Drop if exists
    df_metadata.drop(columns="Unnamed: 0", inplace=True, errors="ignore")

    # Add ml_id column if not in metadata already
    if "ml_id" not in df_metadata.columns:
        df_metadata["ml_id"] = -1.0

    _log.info(f"Found {len(df_metadata)} Wind systems in {filename}")

    return df_metadata


def load_everything_into_ram(
    generation_filename,
    metadata_filename,
    estimated_capacity_percentile: float = 100,
) -> xr.DataArray:
    """Load Wind data into xarray DataArray in RAM.

    Args:
        generation_filename: Filepath to the Wind generation data
        metadata_filename: Filepath to the Wind metadata
        estimated_capacity_percentile: Percentile used as the estimated capacity for each PV
            system. Recommended range is 99-100.
    """

    # load metadata
    df_metadata = _load_wind_metadata(metadata_filename)
    # Load pd.DataFrame of power and pd.Series of capacities:
    df_gen, estimated_capacities = _load_wind_generation_and_capacity(
        generation_filename,
        estimated_capacity_percentile=estimated_capacity_percentile,
    )
    # Drop systems where all values are NaN
    df_gen.dropna(axis="columns", how="all", inplace=True)
    estimated_capacities = estimated_capacities[df_gen.columns]

    # Ensure systems are consistant between generation data, and metadata
    common_systems = list(np.intersect1d(df_metadata.index, df_gen.columns))
    df_gen = df_gen[common_systems]
    df_metadata = df_metadata.loc[common_systems]
    estimated_capacities = estimated_capacities.loc[common_systems]
    # Compile data into an xarray DataArray
    xr_array = put_wind_data_into_an_xr_dataarray(
        df_gen=df_gen,
        observed_system_capacities=estimated_capacities,
        nominal_system_capacities=df_metadata.capacity_megawatts,
        ml_id=df_metadata.ml_id,
        latitude=df_metadata.latitude,
        longitude=df_metadata.longitude,
    )
    # Sanity checks
    time_utc = pd.DatetimeIndex(xr_array.time_utc)
    assert time_utc.is_monotonic_increasing
    assert time_utc.is_unique

    return xr_array


def _load_wind_generation_and_capacity(
    filename: Union[str, Path],
    estimated_capacity_percentile: float = 99,
) -> tuple[pd.DataFrame, pd.Series]:
    """Load the PV data and estimates the capacity for each PV system.

    The capacity is estimated by taking the max value across all datetimes in the input file.

    Args:
        filename: The filename (netcdf) of the wind data to load
        estimated_capacity_percentile: Percentile used as the estimated capacity for each PV
            system. Recommended range is 99-100.

    Returns:
        DataFrame of power output in watts. Columns are PV systems, rows are datetimes
        Series of PV system estimated capacities in watts
    """

    _log.info(f"Loading wind power data from {filename}.")

    with fsspec.open(filename, mode="rb") as file:
        file_bytes = file.read()

    _log.info("Loaded wind power bytes, now converting to xarray")
    with io.BytesIO(file_bytes) as file:
        ds_gen = xr.load_dataset(file, engine="h5netcdf")

    df_gen = ds_gen.to_dataframe()
    # Fix data types
    df_gen = df_gen.astype(np.float32)
    df_gen.columns = df_gen.columns.astype(np.int64)

    _log.info("Loaded wind PV power data and converting to pandas.")
    estimated_capacities = df_gen.quantile(estimated_capacity_percentile / 100)

    # Remove systems with no generation data
    mask = estimated_capacities > 0
    estimated_capacities = estimated_capacities[mask]
    df_gen = df_gen.loc[:, mask]
    _log.info(
        "After loading:"
        f" {len(df_gen)} wind power datetimes."
        f" {len(df_gen.columns)} wind power wind system IDs."
    )

    # Sanity checks
    assert not df_gen.columns.duplicated().any()
    assert not df_gen.index.duplicated().any()
    assert np.isfinite(estimated_capacities).all()
    assert (estimated_capacities > 0).all()
    assert np.array_equal(df_gen.columns, estimated_capacities.index)

    return df_gen, estimated_capacities
