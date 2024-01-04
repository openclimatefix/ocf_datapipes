""" Util functions for Wind data source"""

import numpy as np
import pandas as pd
import xarray as xr


def put_wind_data_into_an_xr_dataarray(
    df_gen: pd.DataFrame,
    observed_system_capacities: pd.Series,
    nominal_system_capacities: pd.Series,
    ml_id: pd.Series,
    longitude: pd.Series,
    latitude: pd.Series,
) -> xr.DataArray:
    """Convert to an xarray DataArray.

    Args:
        df_gen: pd.DataFrame where the columns are Wind systems (and the column names are ints), and
            the index is UTC datetime
        observed_system_capacities: The max power output observed in the time series for wind system
            in megawatts. Index is wind system IDs
        nominal_system_capacities: The metadata value for each wind system capacities in megawatts
        ml_id: The `ml_id` used to identify each PV system
        longitude: longitude of the locations
        latitude: latitude of the locations
    """
    # Sanity check!
    system_ids = df_gen.columns
    for name, series in (
        ("observed_system_capacities", observed_system_capacities),
        ("nominal_system_capacities", nominal_system_capacities),
        ("ml_id", ml_id),
        ("longitude", longitude),
        ("latitude", latitude),
    ):
        if (series is not None) and (not np.array_equal(series.index, system_ids)):
            raise ValueError(
                f"Index of {name} does not equal {system_ids}. Index is {series.index}"
            )

    data_array = xr.DataArray(
        data=df_gen.values,
        coords=(
            ("time_utc", df_gen.index.values),
            ("wind_system_id", system_ids),
        ),
        name="wind_power_megawatts",
    ).astype(np.float32)

    data_array = data_array.assign_coords(
        observed_capacity_mwp=("wind_system_id", observed_system_capacities),
        nominal_capacity_mwp=("wind_system_id", nominal_system_capacities),
        ml_id=("wind_system_id", ml_id),
        longitude=("wind_system_id", longitude),
        latitude=("wind_system_id", latitude),
    )

    return data_array
