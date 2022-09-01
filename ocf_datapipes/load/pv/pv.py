"""Datapipe and utils to load PV data from NetCDF for training"""
import datetime
import logging
from pathlib import Path
from typing import Optional, Union

import fsspec
import numpy as np
import pandas as pd
import xarray as xr
from torchdata.datapipes import functional_datapipe
from torchdata.datapipes.iter import IterDataPipe

from ocf_datapipes.load.pv.utils import (
    intersection_of_pv_system_ids,
    put_pv_data_into_an_xr_dataarray,
)
from ocf_datapipes.utils.geospatial import lat_lon_to_osgb

_log = logging.getLogger(__name__)


@functional_datapipe("open_pv_netcdf")
class OpenPVFromNetCDFIterDataPipe(IterDataPipe):
    """Datapipe to load NetCDF"""

    def __init__(
        self,
        pv_power_filename: Union[str, Path],
        pv_metadata_filename: Union[str, Path],
    ):
        """
        Datapipe to load PV from NetCDF

        Args:
            pv_power_filename: Filename of the power file
            pv_metadata_filename: Filename of the metadata file
        """
        super().__init__()
        self.pv_power_filename = pv_power_filename
        self.pv_metadata_filename = pv_metadata_filename

    def __iter__(self):
        data: xr.DataArray = load_everything_into_ram(
            self.pv_power_filename, self.pv_metadata_filename
        )
        while True:
            yield data


def load_everything_into_ram(pv_power_filename, pv_metadata_filename) -> xr.DataArray:
    """Open AND load PV data into RAM."""
    # Load pd.DataFrame of power and pd.Series of capacities:
    pv_power_watts, pv_capacity_wp, pv_system_row_number = _load_pv_power_watts_and_capacity_wp(
        pv_power_filename,
    )
    pv_metadata = _load_pv_metadata(pv_metadata_filename)
    # Ensure pv_metadata, pv_power_watts, and pv_capacity_wp all have the same set of
    # PV system IDs, in the same order:
    pv_metadata, pv_power_watts = intersection_of_pv_system_ids(pv_metadata, pv_power_watts)
    pv_capacity_wp = pv_capacity_wp.loc[pv_power_watts.columns]
    pv_system_row_number = pv_system_row_number.loc[pv_power_watts.columns]

    data_in_ram = put_pv_data_into_an_xr_dataarray(
        pv_power_watts=pv_power_watts,
        y_osgb=pv_metadata.y_osgb.astype(np.float32),
        x_osgb=pv_metadata.x_osgb.astype(np.float32),
        capacity_wp=pv_capacity_wp,
        pv_system_row_number=pv_system_row_number,
    )

    # Sanity checks:
    time_utc = pd.DatetimeIndex(data_in_ram.time_utc)
    assert time_utc.is_monotonic_increasing
    assert time_utc.is_unique

    return data_in_ram


def _load_pv_power_watts_and_capacity_wp(
    filename: Union[str, Path],
    start_date: Optional[datetime.datetime] = None,
    end_date: Optional[datetime.datetime] = None,
) -> tuple[pd.DataFrame, pd.Series, pd.Series]:
    """Return pv_power_watts, pv_capacity_wp, pv_system_row_number.

    The capacities and pv_system_row_number are computed across the *entire* dataset,
    and so is independent of the `start_date` and `end_date`. This ensures the PV system
    row number and capacities stay constant across training and validation.
    """

    _log.info(f"Loading solar PV power data from {filename} from {start_date=} to {end_date=}.")

    # Load data in a way that will work in the cloud and locally:
    with fsspec.open(filename, mode="rb") as file:
        pv_power_ds = xr.open_dataset(file, engine="h5netcdf")
        pv_capacity_wp = pv_power_ds.max().to_pandas().astype(np.float32)
        pv_power_watts = pv_power_ds.sel(datetime=slice(start_date, end_date)).to_dataframe()
        pv_power_watts = pv_power_watts.astype(np.float32)
        del pv_power_ds

    if "passiv" not in str(filename):
        _log.warning("Converting timezone. ARE YOU SURE THAT'S WHAT YOU WANT TO DO?")
        pv_power_watts = (
            pv_power_watts.tz_localize("Europe/London").tz_convert("UTC").tz_convert(None)
        )

    pv_capacity_wp.index = [np.int32(col) for col in pv_capacity_wp.index]
    pv_power_watts.columns = pv_power_watts.columns.astype(np.int64)

    # Create pv_system_row_number. We use the index of `pv_capacity_wp` because that includes
    # the PV system IDs for the entire dataset (independent of `start_date` and `end_date`).
    # We use `float32` for the ID because we use NaN to indicate a missing PV system,
    # or that this whole example doesn't include PV.
    all_pv_system_ids = pv_capacity_wp.index
    pv_system_row_number = np.arange(start=0, stop=len(all_pv_system_ids), dtype=np.float32)
    pv_system_row_number = pd.Series(pv_system_row_number, index=all_pv_system_ids)

    _log.info(
        "After loading:"
        f" {len(pv_power_watts)} PV power datetimes."
        f" {len(pv_power_watts.columns)} PV power PV system IDs."
    )

    pv_power_watts = pv_power_watts.clip(lower=0, upper=5e7)
    pv_power_watts = _drop_pv_systems_which_produce_overnight(pv_power_watts)

    # Resample to 5-minutely and interpolate up to 15 minutes ahead.
    # TODO: Issue #74: Give users the option to NOT resample (because Perceiver IO
    # doesn't need all the data to be perfectly aligned).
    pv_power_watts = pv_power_watts.resample("5T").interpolate(method="time", limit=3)
    pv_power_watts.dropna(axis="index", how="all", inplace=True)
    pv_power_watts.dropna(axis="columns", how="all", inplace=True)

    # Drop any PV systems whose PV capacity is too low:
    PV_CAPACITY_THRESHOLD_W = 100
    pv_systems_to_drop = pv_capacity_wp.index[pv_capacity_wp <= PV_CAPACITY_THRESHOLD_W]
    pv_systems_to_drop = pv_systems_to_drop.intersection(pv_power_watts.columns)
    _log.info(
        f"Dropping {len(pv_systems_to_drop)} PV systems because their max power is less than"
        f" {PV_CAPACITY_THRESHOLD_W}"
    )
    pv_power_watts.drop(columns=pv_systems_to_drop, inplace=True)

    # Ensure that capacity and pv_system_row_num use the same PV system IDs as the power DF:
    pv_system_ids = pv_power_watts.columns
    pv_capacity_wp = pv_capacity_wp.loc[pv_system_ids]
    pv_system_row_number = pv_system_row_number.loc[pv_system_ids]

    _log.info(
        "After filtering & resampling to 5 minutes:"
        f" pv_power = {pv_power_watts.values.nbytes / 1e6:,.1f} MBytes."
        f" {len(pv_power_watts)} PV power datetimes."
        f" {len(pv_power_watts.columns)} PV power PV system IDs."
    )

    # Sanity checks:
    assert not pv_power_watts.columns.duplicated().any()
    assert not pv_power_watts.index.duplicated().any()
    assert np.isfinite(pv_capacity_wp).all()
    assert (pv_capacity_wp >= 0).all()
    assert np.isfinite(pv_system_row_number).all()
    assert np.array_equal(pv_power_watts.columns, pv_capacity_wp.index)
    return pv_power_watts, pv_capacity_wp, pv_system_row_number


"""Filtering to be added in a different IterDataPipe

    pv_power_watts = pv_power_watts.clip(lower=0, upper=5e7)
    # Convert the pv_system_id column names from strings to ints:
    pv_power_watts.columns = [np.int32(col) for col in pv_power_watts.columns]

    if "passiv" not in filename:
        _log.warning("Converting timezone. ARE YOU SURE THAT'S WHAT YOU WANT TO DO?")
        pv_power_watts = (
            pv_power_watts.tz_localize("Europe/London").tz_convert("UTC").tz_convert(None)
        )

    pv_power_watts = _drop_pv_systems_which_produce_overnight(pv_power_watts)

    # Resample to 5-minutely and interpolate up to 15 minutes ahead.
    # TODO: Issue #74: Give users the option to NOT resample (because Perceiver IO
    # doesn't need all the data to be perfectly aligned).
    pv_power_watts = pv_power_watts.resample("5T").interpolate(method="time", limit=3)
    pv_power_watts.dropna(axis="index", how="all", inplace=True)
    pv_power_watts.dropna(axis="columns", how="all", inplace=True)

    # Drop any PV systems whose PV capacity is too low:
    PV_CAPACITY_THRESHOLD_W = 100
    pv_systems_to_drop = pv_capacity_wp.index[pv_capacity_wp <= PV_CAPACITY_THRESHOLD_W]
    pv_systems_to_drop = pv_systems_to_drop.intersection(pv_power_watts.columns)
    _log.info(
        f"Dropping {len(pv_systems_to_drop)} PV systems because their max power is less than"
        f" {PV_CAPACITY_THRESHOLD_W}"
    )
    pv_power_watts.drop(columns=pv_systems_to_drop, inplace=True)

    # Ensure that capacity and pv_system_row_num use the same PV system IDs as the power DF:
    pv_system_ids = pv_power_watts.columns
    pv_capacity_wp = pv_capacity_wp.loc[pv_system_ids]
    pv_system_row_number = pv_system_row_number.loc[pv_system_ids]

    _log.info(
        "After filtering & resampling to 5 minutes:"
        f" pv_power = {pv_power_watts.values.nbytes / 1e6:,.1f} MBytes."
        f" {len(pv_power_watts)} PV power datetimes."
        f" {len(pv_power_watts.columns)} PV power PV system IDs."
    )


"""


# Adapted from nowcasting_dataset.data_sources.pv.pv_data_source
def _load_pv_metadata(filename: str) -> pd.DataFrame:
    """Return pd.DataFrame of PV metadata.

    Shape of the returned pd.DataFrame for Passiv PV data:
        Index: ss_id (Sheffield Solar ID)
        Columns: llsoacd, orientation, tilt, kwp, operational_at,
            latitude, longitude, system_id, x_osgb, y_osgb
    """
    _log.info(f"Loading PV metadata from {filename}")

    if "passiv" in str(filename):
        index_col = "ss_id"
    else:
        index_col = "system_id"

    pv_metadata = pd.read_csv(filename, index_col=index_col)

    if "Unnamed: 0" in pv_metadata.columns:
        pv_metadata.drop(columns="Unnamed: 0", inplace=True)

    _log.info(f"Found {len(pv_metadata)} PV systems in {filename}")

    # drop any systems with no lon or lat:
    pv_metadata.dropna(subset=["longitude", "latitude"], how="any", inplace=True)

    _log.debug(f"Found {len(pv_metadata)} PV systems with locations")

    pv_metadata["x_osgb"], pv_metadata["y_osgb"] = lat_lon_to_osgb(
        latitude=pv_metadata["latitude"], longitude=pv_metadata["longitude"]
    )

    return pv_metadata


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
