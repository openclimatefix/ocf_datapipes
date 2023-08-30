"""Datapipe and utils to load PV data from NetCDF for training"""
import io
import logging
from datetime import datetime
from pathlib import Path
from typing import List, Optional, Union

import fsspec
import numpy as np
import pandas as pd
import xarray as xr
from torchdata.datapipes import functional_datapipe
from torchdata.datapipes.iter import IterDataPipe

from ocf_datapipes.config.model import PV
from ocf_datapipes.load.pv.utils import (
    put_pv_data_into_an_xr_dataarray,
)

_log = logging.getLogger(__name__)


@functional_datapipe("open_pv_netcdf")
class OpenPVFromNetCDFIterDataPipe(IterDataPipe):
    """Datapipe to load NetCDF"""

    def __init__(
        self,
        pv: PV,
    ):
        """Datapipe to load PV from NetCDF.

        Args:
            pv: pv configuration
        """
        super().__init__()
        self.pv = pv
        self.pv_power_filenames = [
            pv_files_group.pv_filename for pv_files_group in pv.pv_files_groups
        ]
        self.pv_metadata_filenames = [
            pv_files_group.pv_metadata_filename for pv_files_group in pv.pv_files_groups
        ]
        self.inferred_metadata_filenames = [
            pv_files_group.inferred_metadata_filename for pv_files_group in pv.pv_files_groups
        ]
        self.start_datetime = pv.start_datetime
        self.end_datetime = pv.end_datetime

    def __iter__(self):
        pv_datas_xr = []
        for i in range(len(self.pv_power_filenames)):
            one_data: xr.DataArray = load_everything_into_ram(
                self.pv_power_filenames[i],
                self.pv_metadata_filenames[i],
                start_datetime=self.start_datetime,
                end_datetime=self.end_datetime,
                inferred_metadata_filename=self.inferred_metadata_filenames[i],
            )
            pv_datas_xr.append(one_data)

        data = join_pv(pv_datas_xr)

        while True:
            yield data


def join_pv(data_arrays: List[xr.DataArray]) -> xr.DataArray:
    """Join PV data arrays together.

    Args:
        data_arrays: List of PV data arrays

    Returns: one data array containing all pv systems
    """

    if len(data_arrays) == 1:
        return data_arrays[0]

    # expand each dataset to full time_utc
    joined_data_array = xr.concat(data_arrays, dim="pv_system_id")

    return joined_data_array


def load_everything_into_ram(
    generation_filename,
    metadata_filename,
    inferred_metadata_filename: Optional[Union[str, Path]] = None,
    start_datetime: Optional[datetime] = None,
    end_datetime: Optional[datetime] = None,
    estimated_capacity_percentile: float = 100,
) -> xr.DataArray:
    """Load PV data into xarray DataArray in RAM.

    Args:
        generation_filename: Filepath to the PV generation data
        metadata_filename: Filepath to the PV metadata
        inferred_metadata_filename: Filepath to inferred metadata
        start_datetime: Data will be filtered to start at this datetime
        end_datetime: Data will be filtered to end at this datetime
        estimated_capacity_percentile: Percentile used as the estimated capacity for each PV
            system. Recommended range is 99-100.
    """

    # load metadata
    df_metadata = _load_pv_metadata(metadata_filename, inferred_metadata_filename)

    # Load pd.DataFrame of power and pd.Series of capacities:
    df_gen, estimated_capacities = _load_pv_generation_and_capacity(
        generation_filename,
        start_date=start_datetime,
        end_date=end_datetime,
        estimated_capacity_percentile=estimated_capacity_percentile,
    )

    # Drop systems and timestamps where all values are NaN
    df_gen.dropna(axis="index", how="all", inplace=True)
    df_gen.dropna(axis="columns", how="all", inplace=True)
    estimated_capacities = estimated_capacities[df_gen.columns]

    # Ensure systems are consistant between generation data, and metadata
    common_systems = list(np.intersect1d(df_metadata.index, df_gen.columns))
    df_gen = df_gen[common_systems]
    df_metadata = df_metadata.loc[common_systems]
    estimated_capacities = estimated_capacities.loc[common_systems]

    # Compile data into an xarray DataArray
    data_in_ram = put_pv_data_into_an_xr_dataarray(
        df_gen=df_gen,
        system_capacities=estimated_capacities,
        ml_id=df_metadata.ml_id,
        latitude=df_metadata.latitude,
        longitude=df_metadata.longitude,
        tilt=df_metadata.get("tilt"),
        orientation=df_metadata.get("orientation"),
    )

    # Sanity checks
    time_utc = pd.DatetimeIndex(data_in_ram.time_utc)
    assert time_utc.is_monotonic_increasing
    assert time_utc.is_unique

    return data_in_ram


def _load_pv_generation_and_capacity(
    filename: Union[str, Path],
    start_date: Optional[datetime] = None,
    end_date: Optional[datetime] = None,
    estimated_capacity_percentile: float = 99,
) -> tuple[pd.DataFrame, pd.Series]:
    """Load the PV data and estimates the capacity for each PV system.

    The capacity is estimated by taking the max value across all datetimes in the input file.

    Args:
        filename: The filename (netcdf) of the PV data to load
        start_date: Start date to load from
        end_date: End of period to load
        estimated_capacity_percentile: Percentile used as the estimated capacity for each PV
            system. Recommended range is 99-100.

    Returns:
        DataFrame of power output in watts. Columns are PV systems, rows are datetimes
        Series of PV system estimated capacities in watts
    """

    _log.info(f"Loading solar PV power data from {filename} from {start_date=} to {end_date=}.")

    # Load data in a way that will work in the cloud and locally:
    if ".parquet" in str(filename):
        _log.debug(f"Loading PV parquet file {filename}")

        df_raw = pd.read_parquet(filename, engine="fastparquet")

        # Assuming the data is reported 5-minutely
        # The power output in watts is 12 times (60 minutes /5 minutes) the energy in watt-hours
        df_raw["generation_w"] = df_raw["generation_wh"] * 12

        # pivot on ss_id
        df_gen = df_raw.pivot(index="timestamp", columns="ss_id", values="generation_w")
        del df_raw

    else:
        with fsspec.open(filename, mode="rb") as file:
            file_bytes = file.read()

        _log.info("Loaded solar PV power bytes, now converting to xarray")
        with io.BytesIO(file_bytes) as file:
            ds_gen = xr.load_dataset(file, engine="h5netcdf")

        df_gen = ds_gen.to_dataframe()

        if "passiv" not in str(filename):
            _log.warning("Converting timezone. ARE YOU SURE THAT'S WHAT YOU WANT TO DO?")
            try:
                df_gen = df_gen.tz_localize("Europe/London").tz_convert("UTC").tz_convert(None)
            except Exception as e:
                _log.warning(
                    "Could not convert timezone from London to UTC. "
                    "Going to try and carry on anyway"
                )
                _log.warning(e)

    # Fix data types
    df_gen = df_gen.astype(np.float32)
    df_gen.columns = df_gen.columns.astype(np.int64)

    _log.info("Loaded solar PV power data and converting to pandas.")
    estimated_capacities = df_gen.quantile(estimated_capacity_percentile / 100)

    # Filter to given time
    df_gen = df_gen.loc[slice(start_date, end_date)]

    # Remove systems with no generation data
    mask = estimated_capacities > 0
    estimated_capacities = estimated_capacities[mask]
    df_gen = df_gen.loc[:, mask]

    _log.info(
        "After loading:"
        f" {len(df_gen)} PV power datetimes."
        f" {len(df_gen.columns)} PV power PV system IDs."
    )

    # Sanity checks
    assert not df_gen.columns.duplicated().any()
    assert not df_gen.index.duplicated().any()
    assert np.isfinite(estimated_capacities).all()
    assert (estimated_capacities > 0).all()
    assert np.array_equal(df_gen.columns, estimated_capacities.index)

    return df_gen, estimated_capacities


# Adapted from nowcasting_dataset.data_sources.pv.pv_data_source
def _load_pv_metadata(filename: str, inferred_filename: Optional[str] = None) -> pd.DataFrame:
    """Return pd.DataFrame of PV metadata.

    Shape of the returned pd.DataFrame for Passiv PV data:
        Index: ss_id (Sheffield Solar ID)
        Columns: llsoacd, orientation, tilt, kwp, operational_at, latitude, longitude, system_id,
            ml_id
    """
    _log.info(f"Loading PV metadata from {filename}")

    if "passiv" in str(filename):
        index_col = "ss_id"
    else:
        index_col = "system_id"

    df_metadata = pd.read_csv(filename, index_col=index_col)

    # Maybe load inferred metadata if passiv
    if inferred_filename is not None:
        df_metadata = _load_inferred_metadata(filename, df_metadata)

    if "Unnamed: 0" in df_metadata.columns:
        df_metadata.drop(columns="Unnamed: 0", inplace=True)

    # Add ml_id column if not in metadata
    if "ml_id" not in df_metadata.columns:
        df_metadata["ml_id"] = np.nan

    _log.info(f"Found {len(df_metadata)} PV systems in {filename}")

    # Rename PVOutput.org tilt name to be simpler
    # There is a second degree tilt, but this should be fine for now
    if "array_tilt_degrees" in df_metadata.columns:
        df_metadata["tilt"] = df_metadata["array_tilt_degrees"]

    # Need to change orientation to a number if a string (i.e. SE) that PVOutput.org uses by default
    mapping = {
        "S": 180.0,
        "SE": 135.0,
        "SW": 225.0,
        "E": 90.0,
        "W": 270.0,
        "N": 0.0,
        "NE": 45.0,
        "NW": 315.0,
        "EW": np.nan,
    }
    df_metadata = df_metadata.replace({"orientation": mapping})

    return df_metadata


def _load_inferred_metadata(filename: str, df_metadata: pd.DataFrame) -> pd.DataFrame:
    inferred_metadata = pd.read_csv(filename, index_col="ss_id")
    inferred_metadata = inferred_metadata.rename({"capacity": "kwp"})
    # Replace columns with new data if in the PV metadata already
    df_metadata.update(inferred_metadata)
    return df_metadata
