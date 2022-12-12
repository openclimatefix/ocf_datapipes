import logging
from datetime import datetime

import numpy as np
import xarray as xr
from torchdata.datapipes import functional_datapipe
from torchdata.datapipes.iter import IterDataPipe

from ocf_datapipes.utils.utils import datetime64_to_datetime


@functional_datapipe("drop_systems_with_lessthan_oneday_data")
class DropSystemsWithLessThanOneDayDataIterDataPipe(IterDataPipe):
    """Drop systmes with less than one day of data"""

    def __init__(self, source_datapipe: IterDataPipe):
        """
        Remove any PV systems with less than 1 day of data.

        This is done, by counting all the nan values and check the
        count is greater than 289 (number of 5 minute intervals in a day)
        """
        self.source_datapipe = source_datapipe

    def __iter__(self) -> xr.Dataset():
        for xr_dataset in self.source_datapipe:

            dates_list = xr_dataset.coords["time_utc"].values
            dates_length = len(dates_list)
            sysids = list(xr_dataset.keys())

            # sanity check
            assert dates_length % int(289) == 0.0

            just_dates = datetime64_to_datetime(dates_list, just_date=True)
            xr_dataset = xr_dataset.assign_coords(just_date=("time_utc", just_dates))
            xr_dataset = xr_dataset.set_xindex("just_date")
            drop_pv = []
            for key in sysids:
                data = np.asarray(xr_dataset.groupby("just_date")[key].values)
                check = len(data) < 289 or np.isnan(data).all()
                while not check:
                    drop_pv.append(key)
                    break
            for i in drop_pv:
                xr_dataset = xr_dataset.drop(i)
            yield xr_dataset
