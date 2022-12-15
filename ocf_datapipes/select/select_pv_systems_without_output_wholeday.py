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

logger = logging.getLogger(__name__)


@functional_datapipe("select_pv_systems_without_output")
class SelectPVSystemsWithoutOutputIterDataPipe(IterDataPipe):
    """Remove any PV systems with less than 1 day of data.
    
    This is done, by counting all the non values and check the
    count is greater than 289 (number of 5 minute intervals in a day)

    Function to select pv system ids with dates that has no output whatsoever in a given day.
    Returns a two key pair (pvsystem id and date) dictionary with values as the status
    "Active" or "Inactive" in a given day.

    Notes:
        Drop the systems or dates with Inactive
        
    Yield example:
        default dict(<class dict>,{"10003:{"2020-04-01 : "Inactive", "2020-05-01" : "Active"}, "10004": {"2020-04-01" : "Active"}})
        Args:
            source_datapipe: Datapipe of Xarray Dataset emitting timeseries data
    """

    def __init__(self, source_datapipe: IterDataPipe) -> None:

        self.source_datapipe = source_datapipe

    def __iter__(self) -> xr.Dataset():

        for xr_dataset in self.source_datapipe:
            sysid = xr_dataset.coords["pv_system_id"].values
            only_dates = np.asarray(xr_dataset.time_utc.dt.day.values, dtype = int)
            xr_dataset = xr_dataset.assign_coords(only_dates = ("time_utc", only_dates))
            sys_groups = list(xr_dataset.groupby("pv_system_id").groups)

            drop_pv = []
            for i in range(len(sysid)):
                dates = sys_groups[i]

            xr_dataset = xr_dataset.assign_coords(just_date=("datetime", dates_list))
            pvstatus_dict = defaultdict(dict)

            # TODO, think how to do this, not a in 2 loops
            # Still needed to work on this
            for sysid in sysid:
                for date in list(set(dates_list)):
                    xr_array = xr_dataset.groupby("just_date")[date][sysid].values

                    if np.isnan(xr_array).all() or np.all(xr_array == 0) == False:
                        pvstatus = "Active"
                    else:
                        pvstatus = "Inactive"

                    pvstatus_dict[sysid][date] = pvstatus
            xr_dataset = xr_dataset.drop_vars("just_date")

            # sanity check
            assert len(xr_dataset) == len(pvstatus_dict)

            # TODO need to return xarray
            # Still needed to work on this
            yield pvstatus_dict
