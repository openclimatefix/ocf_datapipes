import logging
import random
from datetime import datetime
from typing import Dict, List, Union

import numpy as np
import pandas as pd
import xarray as xr
from torchdata.datapipes import functional_datapipe
from torchdata.datapipes.iter import IterDataPipe

logger = logging.getLogger(__name__)

"""According to the UK Met office- https://www.metoffice.gov.uk/weather/learn-about/met-office-for-schools/other-content/other-resources/our-seasons

Summer  - starts in June in the UK and finishes at the end of August.
Autumn -  starts in September and finishes in November
Winter - runs from December to February
Spring - begins in March and ends in May
According to worlddata.info - https://www.worlddata.info/europe/united-kingdom/sunset.php
Month	    Sunrise	    Sunset      Hours of daylight
January	    07:57 am	04:22 pm	8:25 h
February	07:12 am	05:16 pm	10:05 h
March	    06:12 am	06:06 pm	11:54 h
April	    06:02 am	07:58 pm	13:56 h
May	        05:06 am	08:47 pm	15:41 h
June	    04:40 am	09:21 pm	16:41 h
July	    04:59 am	09:13 pm	16:15 h
August	    05:44 am	08:25 pm	14:41 h
September	06:33 am	07:17 pm	12:44 h
October	    07:22 am	06:09 pm	10:47 h
November	07:16 am	04:13 pm	8:57 h
December	07:57 am	03:53 pm	7:56 h
"""

import logging
from datetime import datetime

import numpy as np
import xarray as xr
from torchdata.datapipes import functional_datapipe
from torchdata.datapipes.iter import IterDataPipe

from ocf_datapipes.utils.utils import datetime64_to_datetime

logger = logging.getLogger(__name__)


@functional_datapipe("drop_night_pv")
class DropNightPVIterDataPipe(IterDataPipe):
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
            
            id_list = xr_dataset.coords["pv_system_id"].values
            groups_idx = xr_dataset.groupby("status_day").groups
            night_idx = groups_idx["night"]
            night_ds = xr_dataset.groupby("status_day")["night"]
            nopvid = []
            for i in id_list:
                data = night_ds.loc[dict(pv_system_id = i)]
                check = np.all(data.values == 0.0) or np.isnan(data.values).all()
                while not check:
                    nopvid.append(i)
                    break
            xr_dataset = xr_dataset.drop_sel(pv_system_id = nopvid)
            yield xr_dataset                

            # TODO in a different PR, try to do this without a loop, this is normally quicker
            # It has been moved to a different .py file : assign_daynight_status.py
            # select only day
            # TODO drop system is producing power over night
            # This method drops the systems producing overnight
