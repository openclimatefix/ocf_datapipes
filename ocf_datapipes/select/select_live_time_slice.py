from torchdata.datapipes.iter import IterDataPipe
from torchdata.datapipes import functional_datapipe
from datetime import timedelta
import pandas as pd
import numpy as np

@functional_datapipe("select_live_t0_time_slice")
class SelectLiveT0TimeSliceIterDataPipe(IterDataPipe):
    """Select the history for the live data"""
    def __init__(self, source_datapipe: IterDataPipe, history_duration: timedelta):
        self.source_datapipe = source_datapipe
        self.history_duration = np.timedelta64(history_duration)

    def __iter__(self):
        for xr_data in self.source_datapipe:
            # Get most recent time in data
            # Select the history that goes back that far
            latest_time_idx = pd.DatetimeIndex(xr_data['time_utc'].values).get_loc(pd.Timestamp.utcnow(), method='pad')
            latest_time = xr_data['time_utc'].values[latest_time_idx]
            xr_data = xr_data.sel(
                time_utc=slice(
                    latest_time - self.history_duration,
                    latest_time,
                )
            )
            yield xr_data
