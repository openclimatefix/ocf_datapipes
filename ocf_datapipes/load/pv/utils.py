""" Util functions for PV data source"""
import logging
from typing import List, Optional

import numpy as np
import pandas as pd
import xarray as xr
from nowcasting_datamodel.models.pv import providers

logger = logging.getLogger(__name__)


def put_pv_data_into_an_xr_dataarray(
    df_gen: pd.DataFrame,
    system_capacities: pd.Series,
    ml_id: pd.Series,
    longitude: pd.Series,
    latitude: pd.Series,
    tilt: Optional[pd.Series] = None,
    orientation: Optional[pd.Series] = None,
) -> xr.DataArray:
    """Convert to an xarray DataArray.

    Args:
        df_gen: pd.DataFrame where the columns are PV systems (and the column names are ints), and
            the index is UTC datetime
        system_capacities: The max power output of each PV system in Watts. Index is PV system IDs.
        ml_id: The `ml_id` used to identify each PV system
        longitude: longitude of the locations
        latitude: latitude of the locations
        tilt: Tilt of the panels
        orientation: Orientation of the panels
    """
    # Sanity check!
    system_ids = df_gen.columns
    for name, series in (
        ("longitude", longitude),
        ("latitude", latitude),
        ("system_capacities", system_capacities),
    ):
        logger.debug(f"Checking {name}")
        if not np.array_equal(series.index, system_ids, equal_nan=True):
            logger.debug(f"Index of {name} does not equal {system_ids}. Index is {series.index}")
            assert np.array_equal(series.index, system_ids, equal_nan=True)

    data_array = xr.DataArray(
        data=df_gen.values,
        coords=(
            ("time_utc", df_gen.index.values),
            ("pv_system_id", system_ids),
        ),
        name="pv_power_watts",
    ).astype(np.float32)

    data_array = data_array.assign_coords(
        longitude=("pv_system_id", longitude),
        latitude=("pv_system_id", latitude),
        capacity_watt_power=("pv_system_id", system_capacities),
        ml_id=("pv_system_id", ml_id),
    )

    if tilt is not None:
        data_array = data_array.assign_coords(
            tilt=("pv_system_id", tilt),
        )
    if orientation is not None:
        data_array = data_array.assign_coords(
            orientation=("pv_system_id", orientation),
        )

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
