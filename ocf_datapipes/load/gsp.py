"""GSP Loader"""
import datetime
import logging
from pathlib import Path
from typing import Optional, Union

import geopandas as gpd
import numpy as np
import pandas as pd
import xarray as xr
from torchdata.datapipes import functional_datapipe
from torchdata.datapipes.iter import IterDataPipe

logger = logging.getLogger(__name__)

try:
    from ocf_datapipes.utils.eso import get_gsp_metadata_from_eso, get_gsp_shape_from_eso

    _has_pvlive = True
except ImportError:
    print("Unable to import PVLive utils, please provide filenames with OpenGSP")
    _has_pvlive = False


@functional_datapipe("open_gsp")
class OpenGSPIterDataPipe(IterDataPipe):
    """Get and open the GSP data"""

    def __init__(
        self,
        gsp_pv_power_zarr_path: Union[str, Path],
        gsp_id_to_region_id_filename: Optional[str] = None,
        sheffield_solar_region_path: Optional[str] = None,
        threshold_mw: int = 0,
        sample_period_duration: datetime.timedelta = datetime.timedelta(minutes=30),
    ):
        """
        Get and open the GSP data

        Args:
            gsp_pv_power_zarr_path: Path to the Zarr for GSP PV Power
            gsp_id_to_region_id_filename: Path to the file containing the mapping of ID ot region
            sheffield_solar_region_path: Path to the Sheffield Solar region data
            threshold_mw: Threshold to drop GSPs by
            sample_period_duration: Sample period of the GSP data
        """
        self.gsp_pv_power_zarr_path = gsp_pv_power_zarr_path
        if (
            gsp_id_to_region_id_filename is None
            or sheffield_solar_region_path is None
            and _has_pvlive
        ):
            self.gsp_id_to_region_id_filename = get_gsp_metadata_from_eso()
            self.sheffield_solar_region_path = get_gsp_shape_from_eso()
        else:
            self.gsp_id_to_region_id_filename = gsp_id_to_region_id_filename
            self.sheffield_solar_region_path = sheffield_solar_region_path
        self.threshold_mw = threshold_mw
        self.sample_period_duration = sample_period_duration

    def __iter__(self) -> xr.DataArray:
        """Get and return GSP data"""
        gsp_id_to_shape = _get_gsp_id_to_shape(
            self.gsp_id_to_region_id_filename, self.sheffield_solar_region_path
        )
        self._gsp_id_to_shape = gsp_id_to_shape  # Save, mostly for plotting to check all is fine!

        logger.debug("Getting GSP data")

        # Load GSP generation xr.Dataset:
        gsp_pv_power_mw_ds = xr.open_dataset(self.gsp_pv_power_zarr_path, engine="zarr")

        # Have to remove ID 0 (National one) for rest to work
        # TODO Do filtering later, deal with national here for now
        gsp_pv_power_mw_ds = gsp_pv_power_mw_ds.isel(
            gsp_id=slice(1, len(gsp_pv_power_mw_ds.gsp_id))
        )

        # Ensure the centroids have the same GSP ID index as the GSP PV power:
        gsp_id_to_shape = gsp_id_to_shape.loc[gsp_pv_power_mw_ds.gsp_id]

        data_array = _put_gsp_data_into_an_xr_dataarray(
            gsp_pv_power_mw=gsp_pv_power_mw_ds.generation_mw.data.astype(np.float32),
            time_utc=gsp_pv_power_mw_ds.datetime_gmt.data,
            gsp_id=gsp_pv_power_mw_ds.gsp_id.data,
            # TODO: Try using `gsp_id_to_shape.geometry.envelope.centroid`. See issue #76.
            x_osgb=gsp_id_to_shape.geometry.centroid.x.astype(np.float32),
            y_osgb=gsp_id_to_shape.geometry.centroid.y.astype(np.float32),
            capacity_megawatt_power=gsp_pv_power_mw_ds.installedcapacity_megawatt_power.data.astype(np.float32),
        )

        del gsp_id_to_shape, gsp_pv_power_mw_ds
        while True:
            yield data_array


def _get_gsp_id_to_shape(
    gsp_id_to_region_id_filename: str, sheffield_solar_region_path: str
) -> gpd.GeoDataFrame:
    """
    Get the GSP ID to the shape

    Args:
        gsp_id_to_region_id_filename: Filename of the mapping file
        sheffield_solar_region_path: Path to the region shaps

    Returns:
        GeoDataFrame containing the mapping from ID to shape
    """
    # Load mapping from GSP ID to Sheffield Solar region ID:
    gsp_id_to_region_id = pd.read_csv(
        gsp_id_to_region_id_filename,
        usecols=["gsp_id", "region_id"],
        dtype={"gsp_id": np.int64, "region_id": np.int64},
    )

    # Load Sheffield Solar region shapes (which are already in OSGB36 CRS).
    ss_regions = gpd.read_file(sheffield_solar_region_path)

    # Merge, so we have a mapping from GSP ID to SS region shape:
    gsp_id_to_shape = (
        ss_regions.merge(gsp_id_to_region_id, left_on="RegionID", right_on="region_id")
        .set_index("gsp_id")[["geometry"]]
        .sort_index()
    )

    # Some GSPs are represented by multiple shapes. To find the correct centroid,
    # we need to find the spatial union of those regions, and then find the centroid
    # of those spatial unions. `dissolve(by="gsp_id")` groups by "gsp_id" and gets
    # the spatial union.
    return gsp_id_to_shape.dissolve(by="gsp_id")


def _put_gsp_data_into_an_xr_dataarray(
    gsp_pv_power_mw: np.ndarray,
    time_utc: np.ndarray,
    gsp_id: np.ndarray,
    x_osgb: np.ndarray,
    y_osgb: np.ndarray,
    capacity_megawatt_power: np.ndarray,
) -> xr.DataArray:
    """
    Converts the GSP data to Xarray DataArray

    Args:
        gsp_pv_power_mw: GSP PV Power
        time_utc: Time in UTC
        gsp_id: Id of the GSPs
        x_osgb: OSGB X coordinates
        y_osgb: OSGB y coordinates
        capacity_megawatt_power: Capacity of each GSP

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
        capacity_megawatt_power=(("time_utc", "gsp_id"), capacity_megawatt_power),
    )
    return data_array
