"""Selects time slice"""
import logging
from datetime import timedelta
from typing import Optional, Union

import numpy as np
import pandas as pd
import xarray as xr
from torchdata.datapipes import functional_datapipe
from torchdata.datapipes.iter import IterDataPipe

logger = logging.getLogger(__name__)


@functional_datapipe("select_time_slice")
class SelectTimeSliceIterDataPipe(IterDataPipe):
    """Selects time slice"""

    def __init__(
        self,
        source_datapipe: IterDataPipe,
        t0_datapipe: IterDataPipe,
        sample_period_duration: timedelta,
        history_duration: Optional[timedelta] = None,
        forecast_duration: Optional[timedelta] = None,
        interval_start: Optional[timedelta] = None,
        interval_end: Optional[timedelta] = None,
        fill_selection: Optional[bool] = False,
    ):
        """
        Selects time slice.

        Either `history_duration` and `history_duration` or `interval_start` and
        `interval_end` must be supplied.

        Args:
            source_datapipe: Datapipe of Xarray objects
            t0_datapipe: Datapipe of t0 times
            sample_period_duration: Sample period of xarray data
            history_duration (optional): History time used
            forecast_duration (optional): Forecast time used
            interval_start (optional): timedelta with respect to t0 where the open interval begins
            interval_end (optional): timedelta with respect to t0 where the open interval ends
            fill_selection (optional): If True, and if the data yielded from `source_datapipe` does
                not extend over the entire requested time period. The missing timestamps are filled
                with NaN values in the returned xarray object. Else the default xarray slicing
                behaviour is used.
        """
        self.source_datapipe = source_datapipe
        self.t0_datapipe = t0_datapipe
        self.fill_selection = fill_selection

        used_duration = history_duration is not None and forecast_duration is not None
        used_intervals = interval_start is not None and interval_end is not None
        assert used_duration ^ used_intervals, "Either durations, or intervals must be supplied"

        if used_duration:
            self.interval_start = -np.timedelta64(history_duration)
            self.interval_end = np.timedelta64(forecast_duration)
        elif used_intervals:
            self.interval_start = np.timedelta64(interval_start)
            self.interval_end = np.timedelta64(interval_end)

        self.sample_period_duration = sample_period_duration

    def _sel_fillnan(self, xr_data, start_dt, end_dt):
        requested_times = pd.date_range(
            start_dt,
            end_dt,
            freq=self.sample_period_duration,
        )
        # Missing time indexes are returned with all NaN values
        return xr_data.reindex(time_utc=requested_times)

    def _sel_default(self, xr_data, start_dt, end_dt):
        return xr_data.sel(time_utc=slice(start_dt, end_dt))

    def __iter__(self) -> Union[xr.DataArray, xr.Dataset]:
        xr_data = next(iter(self.source_datapipe))

        for t0 in self.t0_datapipe:
            t0_datetime_utc = pd.Timestamp(t0)
            start_dt = t0_datetime_utc + self.interval_start
            end_dt = t0_datetime_utc + self.interval_end

            start_dt = start_dt.ceil(self.sample_period_duration)
            end_dt = end_dt.ceil(self.sample_period_duration)

            if self.fill_selection:
                yield self._sel_fillnan(xr_data, start_dt, end_dt)
            else:
                yield self._sel_default(xr_data, start_dt, end_dt)
