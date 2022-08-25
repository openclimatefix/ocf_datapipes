from torchdata.datapipes.iter import IterDataPipe, Zipper
from torchdata.datapipes import functional_datapipe
from typing import Optional

import numpy as np
import pandas as pd
import numpy as np
import xarray as xr
from datetime import timedelta

@functional_datapipe("select_live_time_slice")
class SelectLiveTimeSliceIterDataPipe(IterDataPipe):
    """Select the history for the live data"""
    def __init__(self, source_datapipe: IterDataPipe, t0_datapipe: IterDataPipe, history_duration: timedelta, dim_name: str = "time_utc"):
        self.source_datapipe = source_datapipe
        self.t0_datapipe = t0_datapipe
        self.history_duration = np.timedelta64(history_duration)
        self.dim_name = dim_name

    def __iter__(self):
        for xr_data, t0 in Zipper(self.source_datapipe, self.t0_datapipe):
            xr_data = xr_data.sel({
                self.dim_name: slice(
                    t0 - self.history_duration,
                    t0
                )}
            )
            yield xr_data
