"""Convert NWP data to the target time"""
import logging
from datetime import timedelta
from typing import Union, Optional

import numpy as np
import pandas as pd
import xarray as xr
from torchdata.datapipes import functional_datapipe
from torchdata.datapipes.iter import IterDataPipe

from ocf_datapipes.utils.utils import profile

logger = logging.getLogger(__name__)


@functional_datapipe("convert_to_nwp_target_time_with_dropout")
class ConvertToNWPTargetTimeWithDropoutIterDataPipe(IterDataPipe):
    """Converts NWP Xarray to use the target time"""

    def __init__(
        self,
        source_datapipe: IterDataPipe,
        t0_datapipe: IterDataPipe,
        sample_period_duration: timedelta,
        history_duration: timedelta,
        forecast_duration: timedelta,
        dropout_time_start: timedelta,
        dropout_time_end: timedelta,
        dropout_frac: Optional[float] = 0,
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
            dropout_time_start: Minimum timedelta (negative, inclusive) w.r.t. t0 when dropout could 
                begin
            dropout_time_end: Minimum timedelta (negative, inclusive) w.r.t. t0 when dropout could 
                begin
            dropout_frac: Fraction of samples subject to dropout
        """
        self.source_datapipe = source_datapipe
        self.t0_datapipe = t0_datapipe
        self.sample_period_duration = sample_period_duration
        self.history_duration = history_duration
        self.forecast_duration = forecast_duration
        self.dropout_time_start = dropout_time_start
        self.dropout_time_end = dropout_time_end
        self.dropout_frac = dropout_frac
        assert dropout_frac >= 0
        assert dropout_time_start <= dropout_time_end
        assert dropout_time_start%sample_period_duration==timedelta(minutes=0)
        assert dropout_time_end%sample_period_duration==timedelta(minutes=0)

    def __len__(self):
        return len(self.t0_datapipe)
        
    def __iter__(self) -> Union[xr.DataArray, xr.Dataset]:
        """Iterate through both datapipes and convert Xarray dataset"""
        
        xr_data = next(iter(self.source_datapipe))
        
        timedeltas = np.arange(
            self.dropout_time_start, 
            self.dropout_time_end+self.sample_period_duration, 
            self.sample_period_duration,
        )
        
        for t0 in self.t0_datapipe:

            with profile("convert_to_nwp_target_time_with_dropout"):

                t0 = pd.Timestamp(t0)
                start_dt = t0 - self.history_duration
                end_dt = t0 + self.forecast_duration
                
                target_times = pd.date_range(
                    start_dt.ceil(self.sample_period_duration),
                    end_dt.ceil(self.sample_period_duration),
                    freq=self.sample_period_duration,
                )
                
                # Apply NWP dropout
                if np.random.uniform() < self.dropout_frac:
                    dt = np.random.choice(timedeltas)
                    t0 = t0 + dt - timedelta(seconds=1)
                
                # Forecasts made before t0 (t<t0) 
                xr_available = xr_data.sel(
                        init_time_utc=slice(None, t0)
                )

                init_times = xr_available.sel(
                    init_time_utc=target_times, 
                    method="ffill", # forward fill from init times to target times
                ).init_time_utc.values
                
                steps = target_times - init_times

                # We want one timestep for each target_time_hourly (obviously!) If we simply do
                # nwp.sel(init_time=init_times, step=steps) then we'll get the *product* of
                # init_times and steps, which is not what # we want! Instead, we use xarray's
                # vectorized-indexing mode by using a DataArray indexer.  See the last example here:
                # https://docs.xarray.dev/en/latest/user-guide/indexing.html#more-advanced-indexing
                coords = {"target_time_utc": target_times}
                init_time_indexer = xr.DataArray(init_times, coords=coords)
                step_indexer = xr.DataArray(steps, coords=coords)
                xr_sel = xr_available.sel(step=step_indexer, init_time_utc=init_time_indexer)
            
            yield xr_sel