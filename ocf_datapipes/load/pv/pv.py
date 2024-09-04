"""Datapipe and utils to load PV data from NetCDF for training"""

import io
import logging
from pathlib import Path
from typing import Optional, Union

import fsspec
import numpy as np
import pandas as pd
import xarray as xr
from torch.utils.data import IterDataPipe, functional_datapipe

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
        self.labels = [pv_files_group.label for pv_files_group in pv.pv_files_groups]

    def __iter__(self):
        pv_array_list = []
        for i in range(len(self.pv_power_filenames)):
            pv_array: xr.DataArray = load_everything_into_ram(
                self.pv_power_filenames[i],
                self.pv_metadata_filenames[i],
                inferred_metadata_filename=self.inferred_metadata_filenames[i],
                label=self.labels[i],
            )
            pv_array_list.append(pv_array)

        pv_array = xr.concat(pv_array_list, dim="pv_system_id")

        while True:
            yield pv_array


def load_everything_into_ram(
    generation_filename,
    metadata_filename,
    inferred_metadata_filename: Optional[Union[str, Path]] = None,
    estimated_capacity_percentile: float = 100,
    label: Optional[str] = None,
) -> xr.DataArray:
    """Load PV data into xarray DataArray in RAM.

    Args:
        generation_filename: Filepath to the PV generation data
        metadata_filename: Filepath to the PV metadata
        inferred_metadata_filename: Filepath to inferred metadata
        estimated_capacity_percentile: Percentile used as the estimated capacity for each PV
            system. Recommended range is 99-100.
        label: Label of which provider the PV data came from
    """

    # load metadata
    df_metadata = _load_pv_metadata(metadata_filename, inferred_metadata_filename, label)

    # Load pd.DataFrame of power and pd.Series of capacities:
    df_gen, estimated_capacities = _load_pv_generation_and_capacity(
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
    xr_array = put_pv_data_into_an_xr_dataarray(
        df_gen=df_gen,
        observed_system_capacities=estimated_capacities,
        nominal_system_capacities=df_metadata.capacity_watts,
        ml_id=df_metadata.ml_id,
        latitude=df_metadata.latitude,
        longitude=df_metadata.longitude,
        tilt=df_metadata.get("tilt"),
        orientation=df_metadata.get("orientation"),
    )

    # Sanity checks
    time_utc = pd.DatetimeIndex(xr_array.time_utc)
    assert time_utc.is_monotonic_increasing
    assert time_utc.is_unique

    return xr_array


def _load_pv_generation_and_capacity(
    filename: Union[str, Path],
    estimated_capacity_percentile: float = 99,
    label: Optional[str] = None,
) -> tuple[pd.DataFrame, pd.Series]:
    """Load the PV data and estimates the capacity for each PV system.

    The capacity is estimated by taking the max value across all datetimes in the input file.

    Args:
        filename: The filename (netcdf) of the PV data to load
        estimated_capacity_percentile: Percentile used as the estimated capacity for each PV
            system. Recommended range is 99-100.
        label: Label of which provider the PV data came from

    Returns:
        DataFrame of power output in watts. Columns are PV systems, rows are datetimes
        Series of PV system estimated capacities in watts
    """

    _log.info(f"Loading solar PV power data from {filename}.")

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

        if label == "pvoutput.org":
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
def _load_pv_metadata(
    filename: str, inferred_filename: Optional[str] = None, label: Optional[str] = None
) -> pd.DataFrame:
    """Return pd.DataFrame of PV metadata.

    Shape of the returned pd.DataFrame for Passiv PV data:
        Index: ss_id (Sheffield Solar ID)
        Columns: llsoacd, orientation, tilt, kwp, operational_at, latitude, longitude, system_id,
            ml_id, capacity_watts
    """
    _log.info(f"Loading PV metadata from {filename}")

    index_col = "ss_id" if label == "solar_sheffield_passiv" else "system_id"
    df_metadata = pd.read_csv(filename, index_col=index_col)

    # Drop if exists
    df_metadata.drop(columns="Unnamed: 0", inplace=True, errors="ignore")

    # Add ml_id column if not in metadata already
    if "ml_id" not in df_metadata.columns:
        df_metadata["ml_id"] = -1.0

    if label == "solar_sheffield_passiv":
        # Add capacity in watts
        df_metadata["capacity_watts"] = df_metadata.kwp * 1000
        # Maybe load inferred metadata if passiv
        if inferred_filename is not None:
            df_metadata = _load_inferred_metadata(filename, df_metadata)
    elif label == "india":  # noqa: E721
        # Add capacity in watts
        df_metadata["capacity_watts"] = df_metadata.capacity_watts
    elif label == "pvoutput.org":
        # For PVOutput.org data
        df_metadata["capacity_watts"] = df_metadata.system_size_watts
        # Rename PVOutput.org tilt name to be simpler
        # There is a second degree tilt, but this should be fine for now
        if "array_tilt_degrees" in df_metadata.columns:
            df_metadata["tilt"] = df_metadata["array_tilt_degrees"]

        # Need to change orientation to a number if a string (i.e. SE) that PVOutput.org uses by
        # default
        mapping = {
            "N": 0.0,
            "NE": 45.0,
            "E": 90.0,
            "SE": 135.0,
            "S": 180.0,
            "SW": 225.0,
            "W": 270.0,
            "NW": 315.0,
        }

        # Any other keys other than those in the dict above are mapped to NaN
        df_metadata["orientation"] = df_metadata.orientation.map(mapping)
    else:
        raise NotImplementedError(f"Provider label {label} not implemented")

    _log.info(f"Found {len(df_metadata)} PV systems in {filename}")

    return df_metadata


def _load_inferred_metadata(filename: str, df_metadata: pd.DataFrame) -> pd.DataFrame:
    inferred_metadata = pd.read_csv(filename, index_col="ss_id")
    inferred_metadata = inferred_metadata.rename({"capacity": "kwp"})
    # Replace columns with new data if in the PV metadata already
    df_metadata.update(inferred_metadata)
    return df_metadata
