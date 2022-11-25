"""Select the history for the live data"""
import logging
from datetime import timedelta
from typing import Union

import numpy as np
import xarray as xr
from torchdata.datapipes import functional_datapipe
from torchdata.datapipes.iter import IterDataPipe

from ocf_datapipes.utils import Zipper

logger = logging.getLogger(__name__)


@functional_datapipe("select_live_time_slice")
class SelectLiveTimeSliceIterDataPipe(IterDataPipe):
    """Select the history for the live data"""

    def __init__(
        self,
        source_datapipe: IterDataPipe,
        t0_datapipe: IterDataPipe,
        history_duration: timedelta,
        dim_name: str = "time_utc",
    ):
        """
        Select the history for the live time slice

        Args:
            source_datapipe: Datapipe emitting Xarray object
            t0_datapipe: Datapipe emitting t0 timestamps
            history_duration: Amount of time for the history
            dim_name: Time dimension name
        """
        self.source_datapipe = source_datapipe
        self.t0_datapipe = t0_datapipe
        self.history_duration = np.timedelta64(history_duration)
        self.dim_name = dim_name

    def __iter__(self) -> Union[xr.DataArray, xr.Dataset]:
        """Select the recent live data"""
        for xr_data, t0 in Zipper(self.source_datapipe, self.t0_datapipe):

            logger.debug(f"Selecting time slice {t0} on dim {self.dim_name}")

            xr_data = xr_data.sel({self.dim_name: slice(t0 - self.history_duration, t0)})

            logger.debug(f"Took slice of length {len(getattr(xr_data,self.dim_name))}")

            yield xr_data
