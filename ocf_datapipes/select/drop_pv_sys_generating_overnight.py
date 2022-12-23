#!/usr/bin/env python3
# -*- coding: UTF-8 -*-

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

            # Collecting all pv system IDs

            id_list = xr_dataset.coords["pv_system_id"].values

            night_ds = xr_dataset.groupby("status_daynight")["night"]
            logger.info(f"Grouping the Xarray by status{night_ds}")

            # Checking if the night time has any pv output
            # if so, stroing the pv system IDs to drop

            nopvid = []
            for i in id_list:
                data = night_ds.loc[dict(pv_system_id=i)]
                check = np.all(data.values == 0.0)
                while not check:
                    nopvid.append(i)
                    break

            logger.info(f"Dropping the pv systems{nopvid} with night time pv output")
            xr_dataset = xr_dataset.drop_sel(pv_system_id=nopvid)
            yield xr_dataset
