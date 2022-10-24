"""GSP Loader"""
import datetime
import logging
from pathlib import Path
from typing import Union

import xarray as xr
from torchdata.datapipes import functional_datapipe
from torchdata.datapipes.iter import IterDataPipe

logger = logging.getLogger(__name__)


@functional_datapipe("open_gsp_national")
class OpenGSPNationalIterDataPipe(IterDataPipe):
    """Get and open the GSP data"""

    def __init__(
        self,
        gsp_pv_power_zarr_path: Union[str, Path],
        sample_period_duration: datetime.timedelta = datetime.timedelta(minutes=30),
    ):
        """
        Get and open the GSP data

        Args:
            gsp_pv_power_zarr_path: Path to the Zarr for GSP PV Power
            sample_period_duration: Sample period of the GSP data
        """
        self.gsp_pv_power_zarr_path = gsp_pv_power_zarr_path
        self.sample_period_duration = sample_period_duration

    def __iter__(self) -> xr.DataArray:
        """Get and return GSP data"""

        logger.debug("Getting GSP data")

        # Load GSP generation xr.Dataset:
        gsp_pv_power_mw_ds = xr.load_dataset(self.gsp_pv_power_zarr_path, engine="zarr")

        # only select national data
        logger.debug("Selecting National data only")
        gsp_pv_power_mw_ds = gsp_pv_power_mw_ds.sel(gsp_id=0)

        # rename some variables
        data_array = gsp_pv_power_mw_ds.rename({"datetime_gmt": "time_utc"})
        data_array = data_array.rename({"generation_mw": "gsp_pv_power_mw"})
        data_array = data_array.rename({"installedcapacity_mwp": "capacity_megawatt_power"})

        while True:
            yield data_array
