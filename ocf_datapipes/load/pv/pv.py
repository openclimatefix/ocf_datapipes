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
from ocf_datapipes.utils.geospatial import lat_lon_to_osgb

_log = logging.getLogger(__name__)


#@functional_datapipe("open_pv_netcdf")
class OpenPVFromNetCDFIterDataPipe(IterDataPipe):
    """Datapipe to load NetCDF"""

    def __init__(
        self,
        pv: PV,
    ):
        """
        Datapipe to load PV from NetCDF

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
        self.start_dateime = pv.start_datetime
        self.end_datetime = pv.end_datetime

    def __iter__(self):
        pv_datas_xr = []
        for i in range(len(self.pv_power_filenames)):
            one_data: xr.DataArray = load_everything_into_ram(
                self.pv_power_filenames[i],
                self.pv_metadata_filenames[i],
                start_dateime=self.start_dateime,
                end_datetime=self.end_datetime,
                time_resolution_minutes=self.pv.time_resolution_minutes,
                inferred_metadata_filename=self.inferred_metadata_filenames[i],
            )
            pv_datas_xr.append(one_data)

        data = join_pv(pv_datas_xr)

        while True:
            yield data


def join_pv(data_arrays: List[xr.DataArray]) -> xr.DataArray:
    """
    Join PV data arrays together

    Args:
        data_arrays: the pv data arrays

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
    start_dateime: Optional[datetime] = None,
    end_datetime: Optional[datetime] = None,
    time_resolution_minutes: Optional[int] = 5,
    inferred_metadata_filename: Optional[Union[str, Path]] = None,
) -> xr.DataArray:
    """Open AND load PV data into RAM."""

    # load metadata
    df_metadata = _load_pv_metadata(metadata_filename, inferred_metadata_filename)

    # Load pd.DataFrame of power and pd.Series of capacities:
    df_gen, estimated_capacities = _load_pv_generation_and_capacity(
        generation_filename,
        start_date=start_dateime,
        end_date=end_datetime,
    )
    
    # Apply clip, filter, and resample
    df_gen, estimated_capacities = _clip_filter_and_resample(
        df_gen, 
        estimated_capacities,
        time_resolution_minutes=time_resolution_minutes,
        drop_overnight = True,
    )
    
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

    # Sanity checks:
    time_utc = pd.DatetimeIndex(data_in_ram.time_utc)
    assert time_utc.is_monotonic_increasing
    assert time_utc.is_unique

    return data_in_ram


def _load_pv_generation_and_capacity(
    filename: Union[str, Path],
    start_date: Optional[datetime] = None,
    end_date: Optional[datetime] = None,
) -> tuple[pd.DataFrame, pd.Series]:
    """Load the PV data and estimates the capacity for each PV system.
    
    The capacity is estimated by taking the max value across all datetimes in the input file.
    
    Args:
        filename: The filename (netcdf) of the PV data to load.
        start_date: Start date to load from.
        end_date: End of period to load.

    Returns:
        DataFrame of power output in Watts. Columns are PV systems, rows are datetimes.
        Series of PV system estimated capacities in Watts
    """

    _log.info(f"Loading solar PV power data from {filename} from {start_date=} to {end_date=}.")


    with fsspec.open(filename, mode="rb") as file:
        file_bytes = file.read()

    _log.info("Loaded solar PV power bytes, now converting to xarray")
    with io.BytesIO(file_bytes) as file:
        ds_gen = xr.load_dataset(file, engine="h5netcdf")

    _log.info("Loaded solar PV power data and converting to pandas.")
    estimated_capacities = ds_gen.max().to_pandas().astype(np.float32)
    estimated_capacities.index = estimated_capacities.index.astype(np.int64)
    
    df_gen = ds_gen.sel(datetime=slice(start_date, end_date)).to_dataframe()
    df_gen = df_gen.astype(np.float32)
    df_gen.columns = df_gen.columns.astype(np.int64)

    if "passiv" not in str(filename):
        _log.warning("Converting timezone. ARE YOU SURE THAT'S WHAT YOU WANT TO DO?")
        try:
            df_gen = (
                df_gen.tz_localize("Europe/London").tz_convert("UTC").tz_convert(None)
            )
        except Exception as e:
            _log.warning(
                "Could not convert timezone from London to UTC. "
                "Going to try and carry on anyway"
            )
            _log.warning(e)    

    _log.info(
        "After loading:"
        f" {len(df_gen)} PV power datetimes."
        f" {len(df_gen.columns)} PV power PV system IDs."
    )
    
    # Sanity checks:
    assert not df_gen.columns.duplicated().any()
    assert not df_gen.index.duplicated().any()
    assert np.isfinite(estimated_capacities).all()
    assert (estimated_capacities >= 0).all()
    assert np.array_equal(df_gen.columns, estimated_capacities.index)
    
    return df_gen, estimated_capacities

def _clip_filter_and_resample(
    df_gen, 
    estimated_capacities,
    time_resolution_minutes: Optional[int] = 5,
    drop_overnight = True,
):
    """Clip, filter, and resample the PV data.
    """
    df_gen = df_gen.clip(lower=0, upper=5e7)
    if drop_overnight:
        df_gen = _drop_pv_systems_which_produce_overnight(df_gen)
        estimated_capacities = estimated_capacities[df_gen.columns]

    # Resample to 5-minutely and interpolate up to 15 minutes ahead.
    # TODO: Issue #74: Give users the option to NOT resample (because Perceiver IO
    # doesn't need all the data to be perfectly aligned).
    df_gen = df_gen.resample(f"{time_resolution_minutes}T").interpolate(
        method="time", limit=3
    )
    
    df_gen.dropna(axis="index", how="all", inplace=True)
    df_gen.dropna(axis="columns", how="all", inplace=True)
    estimated_capacities = estimated_capacities[df_gen.columns]

    # Drop any PV systems whose PV capacity is too low:
    CAPACITY_THRESHOLD_W = 100
    mask = (estimated_capacities <= CAPACITY_THRESHOLD_W)
        
    _log.info(
        f"Dropping {mask.sum()} PV systems because their max power is less than"
        f" {CAPACITY_THRESHOLD_W}"
    )
        
    df_gen = df_gen.loc[:, ~mask]
    estimated_capacities = estimated_capacities[~mask]
    
    _log.info(
        f"After filtering & resampling to {time_resolution_minutes} minutes:"
        f" pv_power = {df_gen.values.nbytes / 1e6:,.1f} MBytes."
        f" {len(df_gen)} PV power datetimes."
        f" {len(df_gen.columns)} PV power PV system IDs."
    )

    # Sanity checks:
    assert not df_gen.columns.duplicated().any()
    assert not df_gen.index.duplicated().any()
    assert np.isfinite(estimated_capacities).all()
    assert (estimated_capacities >= 0).all()
    assert np.array_equal(df_gen.columns, estimated_capacities.index), (df_gen.columns, estimated_capacities.index)
    
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
    if "passiv" in str(filename) and inferred_filename is not None:
        df_metadata = _load_inferred_metadata(filename, df_metadata)

    if "Unnamed: 0" in df_metadata.columns:
        df_metadata.drop(columns="Unnamed: 0", inplace=True)
    
    # Add ml_id column if not in metadata
    if "ml_id" not in df_metadata.columns:
        df_metadata["ml_id"] = np.nan 
    df_metadata["ml_id"] = df_metadata.ml_id.astype(pd.Int64Dtype())

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


def _drop_pv_systems_which_produce_overnight(pv_power_watts: pd.DataFrame) -> pd.DataFrame:
    """Drop systems which produce power over night.

    Args:
        pv_power_watts: Un-normalised.
    """
    # TODO: Of these bad systems, 24647, 42656, 42807, 43081, 51247, 59919
    # might have some salvagable data?
    NIGHT_YIELD_THRESHOLD = 0.4
    night_hours = [22, 23, 0, 1, 2]
    pv_power_normalised = pv_power_watts / pv_power_watts.max()
    night_mask = pv_power_normalised.index.hour.isin(night_hours)
    pv_power_at_night_normalised = pv_power_normalised.loc[night_mask]
    pv_above_threshold_at_night = (pv_power_at_night_normalised > NIGHT_YIELD_THRESHOLD).any()
    bad_systems = pv_power_normalised.columns[pv_above_threshold_at_night]
    _log.info(f"{len(bad_systems)} bad PV systems found and removed!")
    return pv_power_watts.drop(columns=bad_systems)


if __name__=="__main__":
    ds = load_everything_into_ram(
        generation_filename="/mnt/disks/nwp/passive/v0/passiv.netcdf",
        metadata_filename="/mnt/disks/nwp/passive/v0/system_metadata_OCF_ONLY.csv",
        start_dateime = "2020-01-01 00:00",
        end_datetime =  "2020-02-01 00:00",
        time_resolution_minutes = 5,
        inferred_metadata_filename = None,
    )
    ds
