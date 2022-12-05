"""Drop PV systems that report overnight"""
import logging
import random
from datetime import datetime
from typing import Dict, List, Union

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


@functional_datapipe("select_pv_systems_with_night_output")
class PVSystemsIDsWithNightOutputIterDataPipe(IterDataPipe):
    """
    Select systems IDs with pv output at the night time
    """

    def __init__(self, source_datapipe: IterDataPipe, meta_datapipe: IterDataPipe):
        """
        This function provides all the system ids which are giving pv output at the night time

        Args
            source_datapipe: A datapipe that emmits Xarray Dataset of the pv.netcdf file
        """
        self.source_datapipe = source_datapipe
        self.meta_datapipe = meta_datapipe

    def __iter__(self) -> pd.DataFrame():
        """Function provides ssid's with night time pv output"""
        # Classifying the night time
        for xr_dataset in self.source_datapipe:
            ssid = random.choice(list(xr_dataset))
            df = xr_dataset[ssid].to_dataframe()
            df["year"] = df.index.year.astype(int)
            df["month"] = df.index.month.astype(int)
            df["day"] = df.index.day.astype(int)
            df["time"] = df.index.time.astype(datetime)
            df["hr"] = df.index.strftime("%H").astype(int)
            df["status"] = df[["month", "hr"]].apply(
                lambda x: "night"  # condtion based on the above data from the comments
                if (x.month == 1 and (x.hr > 17 and x.hr < 7))
                else "day"
                if (x.month == 2) and (x.hr > 18 and x.hr < 7)
                else "day"
                if (x.month == 3) and (x.hr > 19 and x.hr < 6)
                else "day"
                if (x.month == 4) and (x.hr > 20 and x.hr < 6)
                else "day"
                if (x.month == 5) and (x.hr > 21 and x.hr < 5)
                else "day"
                if (x.month == 6) and (x.hr > 22 and x.hr < 4)
                else "day"
                if (x.month == 7) and (x.hr > 22 and x.hr < 5)
                else "day"
                if (x.month == 8) and (x.hr > 21 and x.hr < 5)
                else "day"
                if (x.month == 9) and (x.hr > 20 and x.hr < 6)
                else "day"
                if (x.month == 10) and (x.hr > 21 and x.hr < 7)
                else "day"
                if (x.month == 11) and (x.hr > 17 and x.hr < 7)
                else "day"
                if (x.month == 12) and (x.hr > 17 and x.hr < 7)
                else "day"
            )
