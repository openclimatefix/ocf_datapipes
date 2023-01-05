"""
This is a class function that assigns day night status
"""
import logging

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
logger.info(
    f"The day and night standard hours are set by {'https://www.timeanddate.com/sun/uk/london'}"
)


@functional_datapipe("assign_daynight_status")
class AssignDayNightStatusIterDataPipe(IterDataPipe):
    """Adds a new dimension of day/night status"""

    def __init__(self, source_datapipe: IterDataPipe):
        """This method adds extra coordinate of day night status.

        Args:
            source_datapipe: Datapipe emiiting Xarray Dataset
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

        # Reading the Xarray dataset
        for xr_dataset in self.source_datapipe:

            # Getting Month and Hour values from time_utc and stacking them
            date_month = xr_dataset.time_utc.dt.month.values
            date_hr = xr_dataset.time_utc.dt.hour.values
            month_hr_stack = np.stack((date_month, date_hr))

            # Getting the status of day/night for each timestamp in the dates'
            day_start = np.asarray([uk_daynight_dict[m][0] for m in month_hr_stack[0]])
            day_end = np.asarray([uk_daynight_dict[m][1] for m in month_hr_stack[0]])
            status_daynight = np.where(
                np.logical_and(month_hr_stack[1] >= day_start, month_hr_stack[1] < day_end),
                "day",
                "night",
            )

            # Assigning a new coordinates of 'status_daynight' in the DataArray
            xr_dataset = xr_dataset.assign_coords(status_daynight=(("time_utc"), status_daynight))
            yield xr_dataset
