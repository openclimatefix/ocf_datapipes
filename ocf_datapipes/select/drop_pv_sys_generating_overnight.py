"""
This is a class function that drops the pv systems with generates power over night.
"""
import logging

import numpy as np
import xarray as xr
from torchdata.datapipes import functional_datapipe
from torchdata.datapipes.iter import IterDataPipe

logger = logging.getLogger(__name__)


@functional_datapipe("drop_night_pv")
class DropPvSysGeneratingOvernightIterDataPipe(IterDataPipe):
    """
    Drop the pv systems which generates power over night date from a timeseries xarray Dataset.
    """

    def __init__(self, source_datapipe: IterDataPipe):
        """
        This method drops the PV systems producing output overnight

        Args:
            source_datapipe: A datapipe that emmits Xarray Dataset of the pv.netcdf file
        """
        self.source_datapipe = source_datapipe

    def __iter__(self) -> xr.DataArray():

        logger.warning("This droping of the nighttime pv is only applicable to the UK PV datasets")

        for xr_dataset in self.source_datapipe:

            # Getting the list of the pv system id
            pv_sys_id_list = xr_dataset.coords["pv_system_id"].values

            # Grouping the datastet with only night status
            night_ds = xr_dataset.groupby("status_daynight")["night"].values

            # Getting the shape of the array
            night_ds_shape = night_ds.shape

            # Checking if all the systems has any values other than zero (including NaN)
            check_nonzero = [(np.array(night_ds[:, m]) != 0) for m in range(night_ds_shape[1])]

            # Checking if all the systems has NaN in their daily outputs
            check_isnan = [
                np.logical_not(np.isnan(night_ds[:, m])) for m in range(night_ds_shape[1])
            ]

            # Checking if there are any systems which has numeric values (not including np.zeros)
            # and (not including np.nan)
            final_check = np.logical_and(check_nonzero, check_isnan)

            # Getting the indices of system ids to drop
            drop_pv_sys_id = np.where(final_check.any(axis=1))[0]

            logger.info(
                f"Dropping the pv systems{pv_sys_id_list[drop_pv_sys_id]} with night time pv output"
            )
            xr_dataset = xr_dataset.drop_sel(pv_system_id=pv_sys_id_list[drop_pv_sys_id])
            yield xr_dataset
