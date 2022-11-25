"""Convert NWP data to the target time"""
import logging
from datetime import timedelta
from typing import Union

import pandas as pd
import xarray as xr
from torchdata.datapipes import functional_datapipe
from torchdata.datapipes.iter import IterDataPipe

logger = logging.getLogger(__name__)


@functional_datapipe("convert_to_nwp_target_time")
class ConvertToNWPTargetTimeIterDataPipe(IterDataPipe):
    """Converts NWP Xarray to use the target time"""

    def __init__(
        self,
        source_datapipe: IterDataPipe,
        t0_datapipe: IterDataPipe,
        sample_period_duration: timedelta,
        history_duration: timedelta,
        forecast_duration: timedelta,
    ):
        """
        Convert NWP Xarray dataset to use target time as indexer

        Args:
            source_datapipe: Datapipe emitting a Xarray Dataset
                with step and init_time_utc indexers
            t0_datapipe: Datapipe emitting t0 times for indexing off of
                choosing the closest previous init_time_utc
            sample_period_duration: How long the sampling period is
            history_duration: How long the history time should cover
            forecast_duration: How long the forecast time should cover
        """
        self.source_datapipe = source_datapipe
        self.t0_datapipe = t0_datapipe
        self.sample_period_duration = sample_period_duration
        self.history_duration = history_duration
        self.forecast_duration = forecast_duration
        self.t0_idx = int(self.history_duration / self.sample_period_duration)

    def __iter__(self) -> Union[xr.DataArray, xr.Dataset]:
        """Iterate through both datapipes and convert Xarray dataset"""
        for xr_data, t0 in self.source_datapipe.zip_ocf(self.t0_datapipe):

            logger.debug("convert_to_nwp_target_time ")

            t0_datetime_utc = pd.Timestamp(t0)
            start_dt = t0_datetime_utc - self.history_duration
            end_dt = t0_datetime_utc + self.forecast_duration
            target_times = pd.date_range(
                start_dt.ceil(self.sample_period_duration),
                end_dt.ceil(self.sample_period_duration),
                freq=self.sample_period_duration,
            )
            # Get the most recent NWP initialisation time for each target_time_hourly.
            init_times = xr_data.sel(init_time_utc=target_times, method="pad").init_time_utc.values
            # Find the NWP init time for just the 'future' portion of the example.
            init_time_t0 = init_times[self.t0_idx]

            # For the 'future' portion of the example, replace all the NWP
            # init times with the NWP init time most recent to t0.
            init_times[self.t0_idx :] = init_time_t0

            steps = target_times - init_times

            # We want one timestep for each target_time_hourly (obviously!) If we simply do
            # nwp.sel(init_time=init_times, step=steps) then we'll get the *product* of
            # init_times and steps, which is not what # we want! Instead, we use xarray's
            # vectorized-indexing mode by using a DataArray indexer.  See the last example here:
            # https://docs.xarray.dev/en/latest/user-guide/indexing.html#more-advanced-indexing
            coords = {"target_time_utc": target_times}
            init_time_indexer = xr.DataArray(init_times, coords=coords)
            step_indexer = xr.DataArray(steps, coords=coords)
            xr_data = xr_data.sel(step=step_indexer, init_time_utc=init_time_indexer)
            yield xr_data
