import logging
from datetime import datetime
from typing import Optional

import numpy as np
import xarray as xr
from torchdata.datapipes import functional_datapipe
from torchdata.datapipes.iter import IterDataPipe

logger = logging.getLogger(__name__)

uk_daynight_dict = {
    1: [7, 17],
    2: [7, 18],
    3: [9, 19],
    4: [6, 20],
    5: [5, 21],
    6: [4, 22],
    7: [5, 22],
    8: [5, 21],
    9: [6, 20],
    10: [7, 21],
    11: [7, 17],
    12: [7, 17],
}
logger.info(f'The day and night standard hours are set by https://www.timeanddate.com/sun/uk/london, {uk_daynight_dict}')

@functional_datapipe("assign_daynight_status")
class AssignDayNightStatusIterDataPipe(IterDataPipe):
    """Adds a new dimension of day/night status"""

    def __init__(self, source_datapipe: IterDataPipe):
        """
        This method adds extra coordinate of day night status.

        Args:
            source: Datapipe emiiting Xarray Dataset

        Result:
            <xarray.Dataset>
            Dimensions:   (datetime: 289)
            Coordinates:
            * datetime  (datetime) datetime64[ns]
            * status_day (datetime)
        """

        self.source_datapipe = source_datapipe

    def __iter__(self) -> xr.DataArray():
        """Returns an xarray dataset with extra dimesion"""
        logger.info(f"Reading the Xarray dataset")
        for xr_dataset in self.source_datapipe:

            logger.info(f"Getting all the {'time_utc'} datetime coordinates")

            dates = xr_dataset.coords["time_utc"].values
            
            logger.info(f"Getting Month and Hour values from {'time_utc'} and stacking them")

            date_month = np.asarray(xr_dataset.time_utc.dt.month.values, dtype=int)
            date_hr = np.asarray(xr_dataset.time_utc.dt.hour.values, dtype=int)
            month_hr_stack = np.stack((date_month, date_hr))

            logger.info(f"Getting the status of day/night for each timestamp in the timeseries")

            status_day = []
            for i in range(len(dates)):
                if month_hr_stack[1][i] in range(
                    uk_daynight_dict[month_hr_stack[0][i]][0],
                    uk_daynight_dict[month_hr_stack[0][i]][1],
                ):
                    status = "day"
                else:
                    status = "night"

                status_day.append(status)

            logger.info(f"Assigning a new coordinates of {'status_day'} in the DataArray")
            
            xr_dataset = xr_dataset.assign_coords(status_day=(("time_utc"), status_day))
            yield xr_dataset
