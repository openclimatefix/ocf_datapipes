"""Selects time slice"""
from datetime import timedelta
from typing import Union

import numpy as np
import pandas as pd
import xarray as xr
from torchdata.datapipes import functional_datapipe
from torchdata.datapipes.iter import IterDataPipe, Zipper


@functional_datapipe("select_time_slice")
class SelectTimeSliceIterDataPipe(IterDataPipe):
    """Selects time slice"""

    def __init__(
        self,
        source_datapipe: IterDataPipe,
        t0_datapipe: IterDataPipe,
        history_duration: timedelta,
        forecast_duration: timedelta,
        sample_period_duration: timedelta,
    ):
        """
        Selects time slice

        Args:
            source_datapipe: Datapipe of Xarray objects
            t0_datapipe: Datapipe of t0 times
            history_duration: History time used
            forecast_duration: Forecast time used
            sample_period_duration: Sample period of xarray data
        """
        self.source_datapipe = source_datapipe
        self.t0_datapipe = t0_datapipe
        self.history_duration = np.timedelta64(history_duration)
        self.forecast_duration = np.timedelta64(forecast_duration)
        self.sample_period_duration = sample_period_duration

    def __iter__(self) -> Union[xr.DataArray, xr.Dataset]:
        for xr_data, t0 in Zipper(self.source_datapipe, self.t0_datapipe):
            t0_datetime_utc = pd.Timestamp(t0)
            start_dt = t0_datetime_utc - self.history_duration
            end_dt = t0_datetime_utc + self.forecast_duration
            xr_data = xr_data.sel(
                time_utc=slice(
                    start_dt.ceil(self.sample_period_duration),
                    end_dt.ceil(self.sample_period_duration),
                )
            )
            yield xr_data
