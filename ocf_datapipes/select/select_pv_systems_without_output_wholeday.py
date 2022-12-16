"""Select PV systems and Dates with No output for the entire day"""
import logging
from collections import defaultdict
from datetime import datetime
from typing import Dict, List, Union

import numpy as np
import pandas as pd
import xarray as xr
from ocf_datapipes.utils.utils import return_sys_idx_with_cont_nan as sys_idx_cont_nan
from torchdata.datapipes import functional_datapipe
from torchdata.datapipes.iter import IterDataPipe

logger = logging.getLogger(__name__)


@functional_datapipe("select_pv_systems_without_output")
class SelectPVSystemsWithoutOutputIterDataPipe(IterDataPipe):
    """Remove any PV systems with less than 1 day of data.

    This is done, by counting all the non values and check the
    count is greater than 289 (number of 5 minute intervals in a day)

        Args:
            source_datapipe: Datapipe of Xarray Dataset emitting timeseries data
    """

    def __init__(self, source_datapipe: IterDataPipe) -> None:

        self.source_datapipe = source_datapipe

    def __iter__(self) -> xr.Dataset():

        for xr_dataset in self.source_datapipe:
            dates_groups = xr_dataset.groupby("only_dates")
            dates = np.asarray(xr_dataset.coords["only_dates"].values, dtype = int)
            dates = np.unique(dates)
            for date in dates:
                xr_ds = dates_groups[date]
                sys_ids = sys_idx_cont_nan(xr_ds.values)
                if not len(sys_ids) == 0.:
                    xr_dataset = xr_dataset.drop_isel(pv_system_id = sysids)
                else:
                    pass
            xr_dataset = xr_dataset.reset_coords("only_dates", drop =True)
            yield xr_dataset
