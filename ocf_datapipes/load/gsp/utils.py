""" Utils for GSP loading"""

from typing import Optional

import geopandas as gpd
import numpy as np
import pandas as pd
import xarray as xr

from ocf_datapipes.utils.location import Location

try:
    from ocf_datapipes.utils.eso import get_gsp_metadata_from_eso, get_gsp_shape_from_eso

    _has_pvlive = True
except ImportError:
    print("Unable to import PVLive utils, please provide filenames with OpenGSP")
    _has_pvlive = False


def put_gsp_data_into_an_xr_dataarray(
    gsp_pv_power_mw: np.ndarray,
    time_utc: np.ndarray,
    gsp_id: np.ndarray,
    x_osgb: np.ndarray,
    y_osgb: np.ndarray,
    nominal_capacity_mwp: np.ndarray,
    effective_capacity_mwp: np.ndarray,
) -> xr.DataArray:
    """
    Converts the GSP data to Xarray DataArray

    Args:
        gsp_pv_power_mw: GSP PV Power
        time_utc: Time in UTC
        gsp_id: Id of the GSPs
        x_osgb: OSGB X coordinates
        y_osgb: OSGB y coordinates
        nominal_capacity_mwp: Installed capacity of each GSP
        effective_capacity_mwp: Estimated effective capacity of each GSP

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
        nominal_capacity_mwp=(("time_utc", "gsp_id"), nominal_capacity_mwp),
        effective_capacity_mwp=(
            ("time_utc", "gsp_id"),
            effective_capacity_mwp,
        ),
    )
    return data_array


def get_gsp_id_to_shape(
    gsp_id_to_region_id_filename: Optional[str] = None,
    sheffield_solar_region_path: Optional[str] = None,
) -> gpd.GeoDataFrame:
    """
    Get the GSP ID to the shape

    Args:
        gsp_id_to_region_id_filename: Filename of the mapping file
        sheffield_solar_region_path: Path to the region shapes

    Returns:
        GeoDataFrame containing the mapping from ID to shape
    """

    did_provide_filepaths = None not in [gsp_id_to_region_id_filename, sheffield_solar_region_path]
    assert _has_pvlive or did_provide_filepaths

    if not did_provide_filepaths:
        if gsp_id_to_region_id_filename is None:
            gsp_id_to_region_id_filename = get_gsp_metadata_from_eso()
        if sheffield_solar_region_path is None:
            sheffield_solar_region_path = get_gsp_shape_from_eso()

    # Load mapping from GSP ID to Sheffield Solar GSP ID to GSP name:
    gsp_id_to_region_id = pd.read_csv(
        gsp_id_to_region_id_filename,
        usecols=["gsp_id", "gsp_name"],
        dtype={"gsp_id": np.int64, "gsp_name": str},
    )

    # Load Sheffield Solar region shapes (which are already in OSGB36 CRS).
    ss_regions = gpd.read_file(sheffield_solar_region_path)

    # Some GSPs are represented by multiple shapes. To find the correct centroid,
    # we need to find the spatial union of those regions, and then find the centroid
    # of those spatial unions. `dissolve(by="GSPs")` groups by "GSPs" and gets
    # the spatial union.
    ss_regions = ss_regions.dissolve(by="GSPs")

    # Merge, so we have a mapping from GSP ID to SS region shape:
    gsp_id_to_shape = (
        ss_regions.merge(gsp_id_to_region_id, left_on="GSPs", right_on="gsp_name")
        .set_index("gsp_id")[["geometry"]]
        .sort_index()
    )

    gsp_0 = (
        gpd.GeoDataFrame(
            data={"gsp_id": [0], "geometry": [gsp_id_to_shape["geometry"].unary_union]}
        )
        .set_index("gsp_id")
        .set_crs(gsp_id_to_shape.crs)
    )

    # For the national forecast, GSP ID 0, we want the shape to be the
    # union of all the other shapes
    gsp_id_to_shape = pd.concat([gsp_id_to_shape, gsp_0]).sort_index()

    # Add central coordinates
    gsp_id_to_shape["x_osgb"] = gsp_id_to_shape.geometry.centroid.x.astype(np.float32)
    gsp_id_to_shape["y_osgb"] = gsp_id_to_shape.geometry.centroid.y.astype(np.float32)

    return gsp_id_to_shape


class GSPLocationLookup:
    """Query object for GSP location from GSP ID"""

    def __init__(
        self,
        gsp_id_to_region_id_filename: Optional[str] = None,
        sheffield_solar_region_path: Optional[str] = None,
    ):
        """Query object for GSP location from GSP ID

        Args:
            gsp_id_to_region_id_filename: Filename of the mapping file
            sheffield_solar_region_path: Path to the region shapes

        """
        self.gsp_id_to_shape = get_gsp_id_to_shape(
            gsp_id_to_region_id_filename,
            sheffield_solar_region_path,
        )

    def __call__(self, gsp_id: int) -> Location:
        """Returns the locations for the input GSP IDs.

        Args:
            gsp_id: Integer ID of the GSP
        """
        return Location(
            x=self.gsp_id_to_shape.loc[gsp_id].x_osgb.astype(np.float32),
            y=self.gsp_id_to_shape.loc[gsp_id].y_osgb.astype(np.float32),
            id=gsp_id,
        )
