import logging
from datetime import datetime
import numpy as np

import xarray as xr
from ocf_datapipes.utils.utils import datetime64_to_datetime
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

@functional_datapipe("assign_daynight_status")
class AssignDayNightStatusIterDataPipe(IterDataPipe):
    """Adds a new dimension of day/night status"""
    
    def __init__(
        self,
        source_datapipe: IterDataPipe):
        """
        This method takes in an numpy.datetime64 values in an Xarray Dataset
        and adds and extra dimesion of day/night status

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
    
    def __iter__(self)-> xr.DataArray():
        """Returns an xarray dataset with extra dimesion"""
        for xr_dataset in self.source_datapipe:
            
            logger.debug(f"Reading the Xarray dataset")
            dates_list = xr_dataset.coords["time_utc"].values
            dtime = datetime64_to_datetime(dates_list)
            date_month = np.asarray([str(i.month) for i in dtime], dtype = int)
            date_hr = np.asarray([str(i.strftime("%H")) for i in dtime], dtype = int)
            month_hr_stack = np.stack((date_month, date_hr))
            status_day = []
            for i in range(len(month_hr_stack[0])):
                if month_hr_stack[1][i] in range(
                    uk_daynight_dict[month_hr_stack[0][i]][0],
                    uk_daynight_dict[month_hr_stack[0][i]][1]
                    ):
                    status = "day"
                else:
                    status = "night"
                status_day.append(status)
            xr_dataset = xr_dataset.assign_coords(status_day = (("time_utc"),status_day))
            xr_dataset = xr_dataset.set_xindex("status_day")
            yield xr_dataset


