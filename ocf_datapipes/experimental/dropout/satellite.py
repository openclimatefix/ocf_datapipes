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

@functional_datapipe("select_dropout_time")
class SelectDropoutTimeIterDataPipe(IterDataPipe):
    """Selects time slice"""

    def __init__(
        self,
        source_datapipe: IterDataPipe,
        dropout_time_start: timedelta,
        dropout_time_end: timedelta,
        dropout_frac: Optional[float] = 0,
        data_pipename: str = None,
        
    ):
        """
        Selects time slice

        Args:
            source_datapipe: Datapipe of t0 times
            dropout_time_start: Minimum timedelta (negative) w.r.t. t0 when dropout could begin
            dropout_time_end: Minimum timedelta (negative) w.r.t. t0 when dropout could begin
            dropout_frac: Fraction of samples subject to dropout
            data_pipename: the name of the data pipe. This is useful when profiling
        """
        self.source_datapipe = source_datapipe
        self.dropout_time_start = dropout_time_start
        self.dropout_time_end = dropout_time_end
        self.dropout_frac = dropout_frac
        self.data_pipename = data_pipename
        assert dropout_frac >= 0
        assert dropout_time_start < dropout_time_end
        
    def __len__(self):
        return len(self.source_datapipe)

    def __iter__(self):
        
        for t0 in self.source_datapipe:
            
            with profile(f"select_dropout_time {self.data_pipename}"):

                t0_datetime_utc = pd.Timestamp(t0)
                
                if np.random.uniform() < self.dropout_frac:
                    dt = np.random.uniform()
                    dt = (
                        dt * self.dropout_time_start +
                        (1-dt)*self.dropout_time_end
                    )
                    dropout_time = t0_datetime_utc + dt
                    
                else:
                    dropout_time = None
            
            yield dropout_time
                    
            

@functional_datapipe("apply_dropout_time")
class ApplyDropoutTimeIterDataPipe(IterDataPipe):
    
    def __init__(
        self,
        source_datapipe: IterDataPipe,
        dropout_time_datapipe: IterDataPipe,
        sample_period_duration: timedelta,
        data_pipename: str = None,
        
    ):
        """

        Args:
            source_datapipe: Datapipe of Xarray objects
            dropout_time_datapipe: Datapipe of dropout times
            sample_period_duration: Sample period of xarray data

            data_pipename: the name of the data pipe. This is useful when profiling
        """
        self.source_datapipe = source_datapipe
        self.dropout_time_datapipe = dropout_time_datapipe
        self.sample_period_duration = sample_period_duration
        self.data_pipename = data_pipename
        
    def __len__(self):
        return len(self.source_datapipe)

    def __iter__(self) -> Union[xr.DataArray, xr.Dataset]:
        
        for xr_data, dropout_time in self.source_datapipe.zip_ocf(self.dropout_time_datapipe):
            
            with profile(f"apply_dropout_time {self.data_pipename}"):

                if dropout_time is None:
                    xr_sel =  xr_data
                
                else:
                    dropout_time = dropout_time.ceil(self.sample_period_duration)
                    
                    # This replaces the times after the dropout with NaNs
                    xr_sel = xr_data.where(xr_data.time_utc<=dropout_time)

            yield xr_sel
