"""Select the history for the live data"""
import pandas as pd
from torch.utils.data import IterDataPipe, functional_datapipe


@functional_datapipe("select_live_t0_time")
class SelectLiveT0TimeIterDataPipe(IterDataPipe):
    """Select the history for the live data"""

    def __init__(self, source_datapipe: IterDataPipe, dim_name: str = "time_utc"):
        """
        Select history for the Xarray object

        Args:
            source_datapipe: Datapipe emitting Xarray objects
            dim_name: The time dimension name to use
        """
        self.source_datapipe = source_datapipe
        self.dim_name = dim_name

    def __iter__(self) -> pd.Timestamp:
        """Get the latest timestamp and return it"""
        for xr_data in self.source_datapipe:
            # Get most recent time in data
            # Select the history that goes back that far
            latest_time_idx = pd.DatetimeIndex(xr_data[self.dim_name].values).get_indexer(
                [pd.Timestamp.now(tz=None)], method="pad"
            )[0]
            latest_time = xr_data[self.dim_name].values[latest_time_idx]
            yield latest_time
