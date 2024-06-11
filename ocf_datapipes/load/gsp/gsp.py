"""GSP Loader"""

import datetime
import logging
from pathlib import Path
from typing import Optional, Union

import numpy as np
import xarray as xr
from torch.utils.data import IterDataPipe, functional_datapipe

from ocf_datapipes.load.gsp.utils import get_gsp_id_to_shape, put_gsp_data_into_an_xr_dataarray

logger = logging.getLogger(__name__)


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

        self.gsp_id_to_region_id_filename = gsp_id_to_region_id_filename
        self.sheffield_solar_region_path = sheffield_solar_region_path
        self.threshold_mw = threshold_mw
        self.sample_period_duration = sample_period_duration

    def __iter__(self) -> xr.DataArray:
        """Get and return GSP data"""
        gsp_id_to_shape = get_gsp_id_to_shape(
            self.gsp_id_to_region_id_filename,
            self.sheffield_solar_region_path,
        )

        logger.debug(f"Getting GSP data from {self.gsp_pv_power_zarr_path}")

        # Load GSP generation xr.Dataset
        gsp_pv_power_mw_ds = xr.open_dataset(self.gsp_pv_power_zarr_path, engine="zarr")

        # Ensure the centroids have the same GSP ID index as the GSP PV power
        gsp_id_to_shape = gsp_id_to_shape.loc[gsp_pv_power_mw_ds.gsp_id]

        data_array = put_gsp_data_into_an_xr_dataarray(
            gsp_pv_power_mw=gsp_pv_power_mw_ds.generation_mw.data.astype(np.float32),
            time_utc=gsp_pv_power_mw_ds.datetime_gmt.data,
            gsp_id=gsp_pv_power_mw_ds.gsp_id.data,
            # TODO: Try using `gsp_id_to_shape.geometry.envelope.centroid`. See issue #76.
            x_osgb=gsp_id_to_shape.x_osgb.astype(np.float32),
            y_osgb=gsp_id_to_shape.y_osgb.astype(np.float32),
            nominal_capacity_mwp=gsp_pv_power_mw_ds.installedcapacity_mwp.data.astype(np.float32),
            effective_capacity_mwp=gsp_pv_power_mw_ds.capacity_mwp.data.astype(np.float32),
        )

        del gsp_id_to_shape, gsp_pv_power_mw_ds
        while True:
            yield data_array
