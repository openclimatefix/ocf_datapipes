"""Remove PV data """

from torchdata.datapipes import functional_datapipe
from torchdata.datapipes.iter import IterDataPipe
from datetime import timedelta
import numpy as np

import logging


logger = logging.getLogger(__name__)


@functional_datapipe("pv_remove_zero_data")
class PVPowerRemoveZeroDataIterDataPipe(IterDataPipe):
    """Compute rolling mean of PV power."""

    def __init__(
        self,
        source_datapipe: IterDataPipe,
        window: timedelta,
    ):

        self.source_datapipe = source_datapipe
        self.window = window

    def __iter__(self):
        for xr_data in self.source_datapipe:

            logger.debug(f"Reducing Date if window ({self.window}) of zeros")

            # get rolling window values
            window_length = int(self.window / xr_data.sample_period_duration)
            resampled = xr_data.rolling(dim={"time_utc": window_length}).max()

            xr_data = xr_data.where(resampled > 0.0)
            yield xr_data
