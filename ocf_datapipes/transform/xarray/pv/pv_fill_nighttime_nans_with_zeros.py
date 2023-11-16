"""Fill nighttime PV with NaNs"""

import logging
from typing import Union

import xarray as xr
from torch.utils.data import IterDataPipe, functional_datapipe

logger = logging.getLogger(__name__)


@functional_datapipe("pv_fill_night_nans")
class PVFillNightNansIterDataPipe(IterDataPipe):
    """Fill nighttime nans with zeros"""

    def __init__(
        self,
        source_datapipe: IterDataPipe,
        daynight_method: str = "elevation",
    ):
        """Fill nighttime NaNs with zeros.

        Args:
            source_datapipe: A datapipe that emits xarray Dataset of PV generation
            daynight_method: Method used to assign datetimes to either 'night' or 'day'. Either
                "simple" or "elevation". See `AssignDayNightStatusIterDataPipe` for details
        """
        self.source_datapipe = source_datapipe

    def __iter__(self) -> Union[xr.DataArray, xr.Dataset]:
        """Run iter"""

        for xr_data in self.source_datapipe.assign_daynight_status():
            # get maks data for nighttime and nans
            is_night = xr_data.status_daynight == "night"
            is_nan = xr_data.isnull()
            should_fill = is_night & is_nan

            # set value
            logger.debug("Setting night NaNs to 0")
            xr_data = xr_data.where(~should_fill, other=0.0)

            while True:
                yield xr_data
