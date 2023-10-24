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
    observed_system_capacities: pd.Series,
    nominal_system_capacities: pd.Series,
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
        observed_system_capacities: The max power output observed in the time series for PV system
            in watts. Index is PV system IDs
        nominal_system_capacities: The metadata value for each PV system capacities in watts
        ml_id: The `ml_id` used to identify each PV system
        longitude: longitude of the locations
        latitude: latitude of the locations
        tilt: Tilt of the panels
        orientation: Orientation of the panels
    """
    # Sanity check!
    system_ids = df_gen.columns
    for name, series in (
        ("observed_system_capacities", observed_system_capacities),
        ("nominal_system_capacities", nominal_system_capacities),
        ("ml_id", ml_id),
        ("longitude", longitude),
        ("latitude", latitude),
        ("tilt", tilt),
        ("orientation", orientation),
    ):
        if (series is not None) and (not np.array_equal(series.index, system_ids)):
            raise ValueError(
                f"Index of {name} does not equal {system_ids}. Index is {series.index}"
            )

    data_array = xr.DataArray(
        data=df_gen.values,
        coords=(
            ("time_utc", df_gen.index.values),
            ("pv_system_id", system_ids),
        ),
        name="pv_power_watts",
    ).astype(np.float32)

    data_array = data_array.assign_coords(
        observed_capacity_wp=("pv_system_id", observed_system_capacities),
        nominal_capacity_wp=("pv_system_id", nominal_system_capacities),
        ml_id=("pv_system_id", ml_id),
        longitude=("pv_system_id", longitude),
        latitude=("pv_system_id", latitude),
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
