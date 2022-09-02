import numpy as np
import xarray as xr


def put_gsp_data_into_an_xr_dataarray(
    gsp_pv_power_mw: np.ndarray,
    time_utc: np.ndarray,
    gsp_id: np.ndarray,
    x_osgb: np.ndarray,
    y_osgb: np.ndarray,
    capacity_mwp: np.ndarray,
) -> xr.DataArray:
    """
    Converts the GSP data to Xarray DataArray

    Args:
        gsp_pv_power_mw: GSP PV Power
        time_utc: Time in UTC
        gsp_id: Id of the GSPs
        x_osgb: OSGB X coordinates
        y_osgb: OSGB y coordinates
        capacity_mwp: Capacity of each GSP

    Returns:
        Xarray DataArray of the GSP data
    """
    # Convert to xr.DataArray:
    data_array = xr.DataArray(
        gsp_pv_power_mw,
        coords=(("time_utc", time_utc), ("gsp_id", gsp_id)),
        name="gsp_pv_power_mw",
    )
    data_array = data_array.assign_coords(
        x_osgb=("gsp_id", x_osgb),
        y_osgb=("gsp_id", y_osgb),
        capacity_mwp=(("time_utc", "gsp_id"), capacity_mwp),
    )
    return data_array