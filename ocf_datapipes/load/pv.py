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

_log = logging.getLogger(__name__)


@functional_datapipe("open_pv_netcdf")
class OpenPVFromNetCDFIterDataPipe(IterDataPipe):
    def __init__(
        self,
        pv_power_filename: str,
        pv_metadata_filename: str,
        n_pv_systems_per_example: int,
        sample_period_duration: datetime.timedelta = datetime.timedelta(minutes=5),
    ):
        super().__init__()
        self.pv_power_filename = pv_power_filename
        self.pv_metadata_filename = pv_metadata_filename
        self.n_pv_systems_per_example = n_pv_systems_per_example
        self.sample_period_duration = sample_period_duration

    def __iter__(self):
        data: xr.DataArray = load_everything_into_ram(
            self.pv_power_filename, self.pv_metadata_filename, self.sample_period_duration
        )
        while True:
            yield data


@functional_datapipe("open_pv_from_db")
class OpenPVFromDBIterDataPipe(IterDataPipe):
    def __init__(self):
        super().__init__()

    def __iter__(self):
        pass


def load_everything_into_ram(
    pv_power_filename, pv_metadata_filename, sample_period_duration
) -> xr.DataArray:
    """Open AND load PV data into RAM."""
    # Load pd.DataFrame of power and pd.Series of capacities:
    pv_power_watts, pv_capacity_wp, pv_system_row_number = _load_pv_power_watts_and_capacity_wp(
        pv_power_filename,
    )
    pv_metadata = _load_pv_metadata(pv_metadata_filename)
    # Ensure pv_metadata, pv_power_watts, and pv_capacity_wp all have the same set of
    # PV system IDs, in the same order:
    pv_metadata, pv_power_watts = _intersection_of_pv_system_ids(pv_metadata, pv_power_watts)
    pv_capacity_wp = pv_capacity_wp.loc[pv_power_watts.columns]
    pv_system_row_number = pv_system_row_number.loc[pv_power_watts.columns]

    data_in_ram = _put_pv_data_into_an_xr_dataarray(
        pv_power_watts=pv_power_watts,
        y_osgb=pv_metadata.y_osgb.astype(np.float32),
        x_osgb=pv_metadata.x_osgb.astype(np.float32),
        capacity_wp=pv_capacity_wp,
        pv_system_row_number=pv_system_row_number,
        sample_period_duration=sample_period_duration,
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

    pv_capacity_wp.index = [np.int32(col) for col in pv_capacity_wp.index]

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

    # Sanity checks:
    assert not pv_power_watts.columns.duplicated().any()
    assert not pv_power_watts.index.duplicated().any()
    assert np.isfinite(pv_capacity_wp).all()
    assert (pv_capacity_wp > 0).all()
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
    pv_metadata = pd.read_csv(filename, index_col="ss_id").drop(columns="Unnamed: 0")
    _log.info(f"Found {len(pv_metadata)} PV systems in {filename}")

    # drop any systems with no lon or lat:
    pv_metadata.dropna(subset=["longitude", "latitude"], how="any", inplace=True)

    _log.debug(f"Found {len(pv_metadata)} PV systems with locations")
    return pv_metadata


def _intersection_of_pv_system_ids(
    pv_metadata: pd.DataFrame, pv_power: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Only pick PV systems for which we have metadata."""
    pv_system_ids = pv_metadata.index.intersection(pv_power.columns)
    pv_system_ids = np.sort(pv_system_ids)
    pv_power = pv_power[pv_system_ids]
    pv_metadata = pv_metadata.loc[pv_system_ids]
    return pv_metadata, pv_power


def _put_pv_data_into_an_xr_dataarray(
    pv_power_watts: pd.DataFrame,
    y_osgb: pd.Series,
    x_osgb: pd.Series,
    capacity_wp: pd.Series,
    pv_system_row_number: pd.Series,
    sample_period_duration: datetime.timedelta,
) -> xr.DataArray:
    """Convert to an xarray DataArray.

    Args:
        pv_power_watts: pd.DataFrame where the columns are PV systems (and the column names are
            ints), and the index is UTC datetime.
        x_osgb: The x location. Index = PV system ID ints.
        y_osgb: The y location. Index = PV system ID ints.
        capacity_wp: The max power output of each PV system in Watts. Index = PV system ID ints.
        pv_system_row_number: The integer position of the PV system in the metadata.
            Used to create the PV system ID embedding.
    """
    # Sanity check!
    pv_system_ids = pv_power_watts.columns
    for series in (x_osgb, y_osgb, capacity_wp, pv_system_row_number):
        assert np.array_equal(series.index, pv_system_ids, equal_nan=True)

    data_array = xr.DataArray(
        data=pv_power_watts.values,
        coords=(("time_utc", pv_power_watts.index), ("pv_system_id", pv_power_watts.columns)),
        name="pv_power_watts",
    ).astype(np.float32)

    data_array = data_array.assign_coords(
        x_osgb=("pv_system_id", x_osgb),
        y_osgb=("pv_system_id", y_osgb),
        capacity_wp=("pv_system_id", capacity_wp),
        pv_system_row_number=("pv_system_id", pv_system_row_number),
    )
    # Sample period duration is required so PVDownsample transform knows by how much
    # to change the pv_t0_idx:
    data_array.attrs["sample_period_duration"] = sample_period_duration
    return data_array
