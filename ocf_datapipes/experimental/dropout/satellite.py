"""Selects time slice"""
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

@functional_datapipe("select_time_slice_with_dropout")
class SelectTimeSliceDropoutIterDataPipe(IterDataPipe):
    """Selects time slice"""

    def __init__(
        self,
        source_datapipe: IterDataPipe,
        t0_datapipe: IterDataPipe,
        history_duration: timedelta,
        forecast_duration: timedelta,
        sample_period_duration: timedelta,
        dropout_frac: Optional[float] = 0,
        dropout_duration_bounds: Optional[list[timedelta, timedelta]] = [None, None],
        data_pipename: str = None,
        
    ):
        """
        Selects time slice

        Args:
            source_datapipe: Datapipe of Xarray objects
            t0_datapipe: Datapipe of t0 times
            history_duration: History time used
            forecast_duration: Forecast time used
            sample_period_duration: Sample period of xarray data
            dropout_frac: Fraction of samples subject to dropout
            dropout_duration_bounds: Times with respect to t0 when data is dropped out. Must be 
                negative.
            data_pipename: the name of the data pipe. This is useful when profiling
        """
        self.source_datapipe = source_datapipe
        self.t0_datapipe = t0_datapipe
        self.history_duration = np.timedelta64(history_duration)
        self.forecast_duration = np.timedelta64(forecast_duration)
        self.sample_period_duration = sample_period_duration
        self.dropout_duration_bounds = dropout_duration_bounds
        self.dropout_frac = dropout_frac
        self.data_pipename = data_pipename
        assert dropout_frac>=0
        if dropout_frac>0:
            assert None not in dropout_duration_bounds
            assert dropout_duration_bounds[0] < timedelta(0), "dropout times must be negative wrt t0"
            assert dropout_duration_bounds[1] < timedelta(0), "dropout times must be negative wrt t0"
            assert dropout_duration_bounds[0] < dropout_duration_bounds[1]


            
        

    def __iter__(self) -> Union[xr.DataArray, xr.Dataset]:
        xr_data = next(iter(self.source_datapipe))
        for t0 in self.t0_datapipe:
            
            with profile(f"select_time_slice_with_dropout {self.data_pipename}"):

                t0_datetime_utc = pd.Timestamp(t0)
                start_dt = t0_datetime_utc - self.history_duration
                end_dt = t0_datetime_utc + self.forecast_duration

                start_dt = start_dt.ceil(self.sample_period_duration)
                end_dt = end_dt.ceil(self.sample_period_duration)
                
                # change to debug
                xr_sel = xr_data.sel(
                    time_utc=slice(
                        start_dt,
                        end_dt,
                    )
                )
                
                if (self.dropout_frac>0) and (np.random.uniform() < self.dropout_frac):
                    dt = np.random.uniform()
                    dt = (
                        dt * self.dropout_duration_bounds[0] +
                        (1-dt)*self.dropout_duration_bounds[0]
                    )
                    dropout_time = t0_datetime_utc + dt
                    dropout_time = dropout_time.ceil(self.sample_period_duration)
                    
                    # This replaces the times after the dropout with NaNs
                    xr_sel = xr_sel.where(xr_sel.time_utc<=dropout_time)


            yield xr_sel
