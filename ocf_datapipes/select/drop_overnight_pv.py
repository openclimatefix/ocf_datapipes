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
class XarrayDatasetAfterNighttimeVarsDroppedIterDataPipe(IterDataPipe):
    """
    Select systems IDs with pv output at the night time
    """

    def __init__(
        self,
        source_datapipe: IterDataPipe,
        meta_datapipe: IterDataPipe):
        """
        This function provides all the system ids which are giving pv output at the night time

        Args
            source_datapipe: A datapipe that emmits Xarray Dataset of the pv.netcdf file
        """
        self.source_datapipe = source_datapipe

    def __iter__(
        self)-> xr.Dataset():
        """Function provides ssid's with night time pv output
        """
        #Classifying the night time
        for xr_dataset in self.source_datapipe:
            dates_list = xr_dataset.coords['datetime'].values

            logger.warning("This dropping of the nighttime pv is only applicable to the UK PV datasets")
            def day_status(
                date_time_of_day: datetime
                ):
                dtime = date_time_of_day
                date_month = dtime.month.astype(int)
                date_hr = dtime.strftime("%H").astype(int)
                status_day = 'night'
                if (date_month == 1 and (date_hr!>17 and date_hr!<7) ):
                    status_day = 'day'
                elif (date_month == 2 and (date_hr!>18 and date_hr!<7)):
                    status_day = 'day'
                elif (date_month == 3 and (date_hr!>19 and date_hr!<6)):
                    status_day = 'day'
                elif (date_month == 4 and (date_hr!>20 and date_hr!<6)):
                    status_day = 'day'
                elif (date_month == 5 and (date_hr!>21 and date_hr!<5)):
                    status_day = 'day'
                elif (date_month == 6 and (date_hr!>22 and date_hr!<4)):
                    status_day = 'day'
                elif (date_month == 7 and (date_hr!>22 and date_hr!<5)):
                    status_day = 'day'
                elif (date_month == 8 and (date_hr!>21 and date_hr!<5)):
                    status_day = 'day'
                elif (date_month == 9 and (date_hr!>20 and date_hr!<6)):
                    status_day = 'day'
                elif (date_month == 10 and (date_hr!>21 and date_hr!<7)):
                    status_day = 'day'
                elif (date_month == 11 and (date_hr!>17 and date_hr!<7)):
                    status_day = 'day'
                elif (date_month == 12 and (date_hr!>17 and date_hr!<7)):
                    status_day = 'day'
                else:
                    logger.debug(f"The datetime {dtime} is not an appropriate date")
                return status_day

            status = [day_status(i) for i in dates_list]
            assert len(dates_list) == len(status)
            xr_dataset = xr_dataset.assign_ccords(daynight_status = ("datetime",status))
            xr_dataset = xr_dataset.where(xr_dataset.daynight_status == "day")
            return xr_dataset
