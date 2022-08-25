from torchdata.datapipes.iter import IterDataPipe
from torchdata.datapipes import functional_datapipe
from typing import Optional
from datetime import timedelta
import pandas as pd
import numpy as np

@functional_datapipe("select_live_t0_time_slice")
class SelectLiveTimeSliceIterDataPipe(IterDataPipe):
    """Select the history for the live data"""
    def __init__(self, source_datapipe: IterDataPipe, history_duration: timedelta, forecast_duration: Optional[timedelta] = None, dim_name: str = "time_utc"):
        self.source_datapipe = source_datapipe
        self.history_duration = np.timedelta64(history_duration)
        self.forecast_duration = forecast_duration if forecast_duration is None else np.timedelta64(forecast_duration)
        self.dim_name = dim_name

    def __iter__(self):
        for xr_data in self.source_datapipe:
            # Get most recent time in data
            # Select the history that goes back that far
            # TODO Add support for NWP, whose time should go into the future, although NWP target time might already do that
            latest_time_idx = pd.DatetimeIndex(xr_data[self.dim_name].values).get_loc(pd.Timestamp.utcnow(), method='pad')
            latest_time = xr_data[self.dim_name].values[latest_time_idx]
            xr_data = xr_data.sel({
                self.dim_name: slice(
                    latest_time - self.history_duration,
                    latest_time if self.forecast_duration is not None else
                    latest_time + self.forecast_duration,
                )}
            )
            yield xr_data
