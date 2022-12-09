"""Select PV systems and Dates with No output for the entire day"""
import logging
from collections import defaultdict
from datetime import datetime
from typing import Dict, List, Union, Optional

import numpy as np
import pandas as pd
import xarray as xr
from torchdata.datapipes import functional_datapipe
from torchdata.datapipes.iter import IterDataPipe

logger = logging.getLogger(__name__)


@functional_datapipe("select_pv_systems_without_output")
class SelectPVSystemsWithoutOutputIterDataPipe(IterDataPipe):
    """Function to select pv system ids with dates that has no output whatsoever in a given day.

    Returns a two key pair (pvsystem id and date) dictionary with values as the status
    "Active" or "Inactive" in a given day.

    Yield example:
        default dict(<class dict>,{"10003:{"2020-04-01 : "Inactive", "2020-05-01" : "Active"},
                                     "10004": {"2020-04-01" : "Active"}})

        Args:
            source_datapipe: Datapipe of Xarray Dataset emitting timeseries data
    """

    def __init__(
        self,
        source_datapipe: IterDataPipe,
        x_dim_name: Optional[str] = "time_utc",
        y_dim_name: Optional[str] = "y_osgb") -> None:

        self.source_datapipe = source_datapipe
        self.x_dim_name = x_dim_name
        self.y_dim_name = y_dim_name

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

                    if (np.isnan(xr_array).all() or np.all(xr_array == 0)) == False:
                        pvstatus = "Active"
                    else:
                        pvstatus = "Inactive"

                    pvstatus_dict[sysid][date] = pvstatus
            xr_dataset = xr_dataset.drop_vars("just_date")

            # sanity check
            assert len(xr_dataset) == len(pvstatus_dict)
            yield pvstatus_dict