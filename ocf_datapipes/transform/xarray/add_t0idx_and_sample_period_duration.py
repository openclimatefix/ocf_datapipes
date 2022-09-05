"""Adds useful t0_idx and sample period attributes"""
import logging
from datetime import timedelta
from typing import Union

import xarray as xr
from torchdata.datapipes import functional_datapipe
from torchdata.datapipes.iter import IterDataPipe

logger = logging.getLogger(__name__)


@functional_datapipe("add_t0_idx_and_sample_period_duration")
class AddT0IdxAndSamplePeriodDurationIterDataPipe(IterDataPipe):
    """Add t0_idx and sample_period_duration attributes to datasets for downstream tasks"""

    def __init__(
        self,
        source_datapipe: IterDataPipe,
        sample_period_duration: timedelta,
        history_duration: timedelta,
    ):
        """
        Adds two attributes, t0_idx, and sample_period_duration for downstream datapipes to use

        Args:
            source_datapipe: Datapipe emitting a Xarray DataSet or DataArray
            sample_period_duration: Time between samples
            history_duration: Amount of history in each example
        """
        self.source_datapipe = source_datapipe
        self.sample_period_duration = sample_period_duration
        self.history_duration = history_duration
        self.t0_idx = int(self.history_duration / self.sample_period_duration)

    def __iter__(self) -> Union[xr.DataArray, xr.Dataset]:
        """Adds the two attributes to the xarray objects and returns them"""
        for xr_data in self.source_datapipe:
            logger.debug("Adding t0 and sample_period_duration to xarray")
            xr_data.attrs["t0_idx"] = self.t0_idx
            xr_data.attrs["sample_period_duration"] = self.sample_period_duration
            yield xr_data
