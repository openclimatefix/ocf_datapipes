""" Util functions for PV data source"""
import logging
from datetime import date, datetime, timedelta
from typing import List, Optional

import numpy as np
import pandas as pd
import xarray as xr
from nowcasting_datamodel.models.pv import providers

logger = logging.getLogger(__name__)


def intersection_of_pv_system_ids(
    pv_metadata: pd.DataFrame, pv_power: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Only pick PV systems for which we have metadata."""
    pv_system_ids = pv_metadata.index.intersection(pv_power.columns)
    pv_system_ids = np.sort(pv_system_ids)
    pv_power = pv_power[pv_system_ids]
    pv_metadata = pv_metadata.loc[pv_system_ids]
    return pv_metadata, pv_power


def put_pv_data_into_an_xr_dataarray(
    pv_power_watts: pd.DataFrame,
    y_osgb: pd.Series,
    x_osgb: pd.Series,
    capacity_watt_power: pd.Series,
    pv_system_row_number: pd.Series,
    longitude: Optional[pd.Series] = None,
    latitude: Optional[pd.Series] = None,
) -> xr.DataArray:
    """Convert to an xarray DataArray.

    Args:
        pv_power_watts: pd.DataFrame where the columns are PV systems (and the column names are
            ints), and the index is UTC datetime.
        x_osgb: The x location. Index = PV system ID ints.
        y_osgb: The y location. Index = PV system ID ints.
        capacity_watt_power: The max power output of each PV system in Watts.
         Index = PV system ID ints.
        pv_system_row_number: The integer position of the PV system in the metadata.
            Used to create the PV system ID embedding.
        longitude: longitude of the locations
        latitude: latitude of the locations
    """
    # Sanity check!
    pv_system_ids = pv_power_watts.columns
    for name, series in (
        ("x_osgb", x_osgb),
        ("y_osgb", y_osgb),
        ("capacity_watt_power", capacity_watt_power),
        ("pv_system_row_number", pv_system_row_number),
    ):
        logger.debug(f"Checking {name}")
        if not np.array_equal(series.index, pv_system_ids, equal_nan=True):
            logger.debug(f"Index of {name} does not equal {pv_system_ids}. Index is {series.index}")
            assert np.array_equal(series.index, pv_system_ids, equal_nan=True)

    data_array = xr.DataArray(
        data=pv_power_watts.values,
        coords=(
            ("time_utc", pv_power_watts.index.values),
            ("pv_system_id", pv_power_watts.columns),
        ),
        name="pv_power_watts",
    ).astype(np.float32)

    data_array = data_array.assign_coords(
        x_osgb=("pv_system_id", x_osgb),
        y_osgb=("pv_system_id", y_osgb),
        capacity_watt_power=("pv_system_id", capacity_watt_power),
        pv_system_row_number=("pv_system_id", pv_system_row_number),
    )
    # Sample period duration is required so PVDownsample transform knows by how much
    # to change the pv_t0_idx:
    if latitude is not None:
        data_array = data_array.assign_coords(
            latitude=("pv_system_id", latitude),
        )
    if longitude is not None:
        data_array = data_array.assign_coords(
            longitude=("pv_system_id", longitude),
        )

    assert len(pv_system_row_number) > 0

    return data_array


def encode_label(indexes: List[str], label: str) -> List[str]:
    """
    Encode the label to a list of indexes.

    The new encoding must be integers and unique.
    It would be useful if the indexes can read and deciphered by humans.
    This is done by times the original index by 10
    and adding 1 for passiv or 2 for other lables

    Args:
        indexes: list of indexes
        label: either 'solar_sheffield_passiv' or 'pvoutput.org'

    Returns: list of indexes encoded by label
    """
    assert label in providers
    # this encoding does work if the number of pv providers is more than 10
    assert len(providers) < 10

    label_index = providers.index(label)
    new_index = [int(int(col) * 10 + label_index) for col in indexes]

    return new_index


def xr_to_df(pv_power: xr.Dataset, ssid: str, date_oi: str) -> pd.DataFrame():
    """Function that gives a pv system df of a single day with its output

    Converts xarray dataset into a pandas dataframe,
    and its values for a single pv systme and a single day

    Args:
        pv_power: pv.netcdf file which contains system is and datetime
                    with corresponding pv output values
        ssid: A single ID of the pv system
        date_oi: Date of Interest
    """
    date_1 = datetime.strptime(date_oi, "%Y-%m-%d")
    on_pv_system = pv_power[ssid].to_dataframe()
    next_day = date_1 + timedelta(days=1)
    on_pv_system = on_pv_system[(on_pv_system.index < next_day) & (on_pv_system.index > date_1)]
    return on_pv_system


def dates_list(pv_power: xr.Dataset) -> List:
    """Provides a list of all the dates in the xr dataset

    Converts dates as coordinates from xarray dataset to a list

    Args:
        pv_power: pv.netcdf file which contains system is and datetime
                    with corresponding pv output values
    """
    dates_lst = pv_power["datetime"].values
    dates_lst = [pd.to_datetime(str(i)) for i in dates_lst]
    dates_lst = [i.strftime("%Y-%m-%d") for i in dates_lst]
    dates_lst = list(set(dates_lst))
    return dates_lst
