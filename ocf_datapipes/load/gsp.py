from torchdata.datapipes.iter import IterDataPipe
from torchdata.datapipes import functional_datapipe
import datetime
import geopandas as gpd
import xarray as xr
import pandas as pd
import numpy as np


@functional_datapipe("open_gsp")
class OpenGSPIterDataPipe(IterDataPipe):
    def __init__(
        self,
        gsp_pv_power_zarr_path: str,
        gsp_id_to_region_id_filename: str,
        sheffield_solar_region_path: str,
        threshold_mw: int = 0,
        sample_period_duration: datetime.timedelta = datetime.timedelta(minutes=30),
    ):
        self.gsp_pv_power_zarr_path = gsp_pv_power_zarr_path
        self.gsp_id_to_region_id_filename = gsp_id_to_region_id_filename
        self.sheffield_solar_region_path = sheffield_solar_region_path
        self.threshold_mw = threshold_mw
        self.sample_period_duration = sample_period_duration

    def __iter__(self):
        gsp_id_to_shape = _get_gsp_id_to_shape(
            self.gsp_id_to_region_id_filename, self.sheffield_solar_region_path
        )
        self._gsp_id_to_shape = gsp_id_to_shape  # Save, mostly for plotting to check all is fine!

        # Load GSP generation xr.Dataset:
        gsp_pv_power_mw_ds = xr.open_dataset(self.gsp_pv_power_zarr_path, engine="zarr")

        # Ensure the centroids have the same GSP ID index as the GSP PV power:
        gsp_id_to_shape = gsp_id_to_shape.loc[gsp_pv_power_mw_ds.gsp_id]

        data_array = _put_gsp_data_into_an_xr_dataarray(
            gsp_pv_power_mw=gsp_pv_power_mw_ds.generation_mw.data.astype(np.float32),
            time_utc=gsp_pv_power_mw_ds.datetime_gmt.data,
            gsp_id=gsp_pv_power_mw_ds.gsp_id.data,
            # TODO: Try using `gsp_id_to_shape.geometry.envelope.centroid`. See issue #76.
            x_osgb=gsp_id_to_shape.geometry.centroid.x.astype(np.float32),
            y_osgb=gsp_id_to_shape.geometry.centroid.y.astype(np.float32),
            capacity_mwp=gsp_pv_power_mw_ds.installedcapacity_mwp.data.astype(np.float32),
            t0_idx=self.t0_idx,
        )

        del gsp_id_to_shape, gsp_pv_power_mw_ds
        while True:
            yield data_array


def _get_gsp_id_to_shape(
    gsp_id_to_region_id_filename: str, sheffield_solar_region_path: str
) -> gpd.GeoDataFrame:
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
    capacity_mwp: np.ndarray,
    t0_idx: int,
) -> xr.DataArray:
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
    data_array.attrs["t0_idx"] = t0_idx
    return data_array
