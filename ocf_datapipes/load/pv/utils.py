""" Util functions for PV data source"""
import logging
from typing import List

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
