"""Select PV systems and Dates with No output for the entire day"""
import logging
from collections import defaultdict
from datetime import datetime
from typing import Dict, List, Union

import numpy as np
import pandas as pd
import xarray as xr
from torchdata.datapipes import functional_datapipe
from torchdata.datapipes.iter import IterDataPipe

from ocf_datapipes.load.pv.utils import dates_list as dt_list
from ocf_datapipes.load.pv.utils import xr_to_df

logger = logging.getLogger(__name__)


@functional_datapipe("select_pv_systems_without_output")
class SelectPVSystemsWithoutOutputIterDataPipe(IterDataPipe):
    """
    Returns a two key pair (pvsystem id and date) dictionary with values as the status
    "Active" or "Inactive" in a given day
    """

    def __init__(self, source_datapipe: IterDataPipe) -> None:
        """
        Args:
            source_datapipe: Datapipe of Xarray Dataset emitting timeseries data
        """

        self.source_datapipe = source_datapipe

    def __iter__(self) -> xr.Dataset():

        for xr_dataset in self.source_datapipe:
            xr_dataset = self.source_datapipe
            dates_list = xr_dataset.coords["datetime"].values
            ssid_list = list(xr_dataset)
            dates_list = [
                datetime.strptime(str(x)[:10], "%Y-%m-%d").strftime("%Y-%m-%d") for x in dates_list
            ]

            # sanity check
            assert len(xr_dataset.coords["datetime"].values) == len(dates_list)

            xr_dataset = xr_dataset.assign_coords(just_date=("datetime", dates_list))
            pvstatus_dict = defaultdict(dict)
            for sysid in ssid_list:
                for date in list(set(dates_list)):
                    xr_array = xr_dataset.groupby("just_date")[date][sysid].values

                    if np.isnan(xr_array).all() and np.all(xr_array == 0) == False:
                        pvstatus = "Active"
                    else:
                        pvstatus = "Inactive"

                    pvstatus_dict[sysid][date] = pvstatus
            xr_dataset = xr_dataset.drop_vars("just_date")

            # sanity check
            assert len(xr_dataset) == len(pvstatus_dict)
            yield pvstatus_dict
