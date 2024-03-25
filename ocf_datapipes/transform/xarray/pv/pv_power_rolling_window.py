"""Compute the rolling mean of PV Power data"""

from typing import Optional, Union

import pandas as pd
import xarray as xr
from torch.utils.data import IterDataPipe, functional_datapipe


@functional_datapipe("pv_power_rolling_window")
class PVPowerRollingWindowIterDataPipe(IterDataPipe):
    """Compute rolling mean of PV power."""

    def __init__(
        self,
        source_datapipe: IterDataPipe,
        window: Union[int, pd.tseries.offsets.DateOffset, pd.core.indexers.objects.BaseIndexer] = 3,
        min_periods: Optional[int] = 2,
        center: bool = True,
        win_type: Optional[str] = None,
        expect_dataset: bool = True,
    ):
        """
        Compute the rolling mean of PV power data

        Args:
            source_datapipe: Datapipe emitting PV Xarray object

            window: Size of the moving window.
            If an integer, the fixed number of observations used for each window.

            If an offset, the time period of each window. Each window will be a variable sized
            based on the observations included in the time-period. This is only valid for
            datetimelike indexes. To learn more about the offsets & frequency strings, please see:
            https://pandas.pydata.org/pandas-docs/stable/user_guide/timeseries.html#offset-aliases

            If a BaseIndexer subclass, the window boundaries based on the defined
            `get_window_bounds` method. Additional rolling keyword arguments,
            namely `min_periods` and `center` will be passed to `get_window_bounds`.

            min_periods: Minimum number of observations in window required to have a value;
            otherwise, result is `np.nan`.

            To avoid NaNs at the start and end of the timeseries, this should be <= ceil(window/2).

            For a window that is specified by an offset, `min_periods` will default to 1.

            For a window that is specified by an integer, `min_periods` will default to the size of
            the window.

            center: If False, set the window labels as the right edge of the window index.
            If True, set the window labels as the center of the window index.

            win_type: Window type
            expect_dataset: Whether to expect a dataset or DataArray
        """
        self.source_datapipe = source_datapipe
        self.window = window
        self.min_periods = min_periods
        self.center = center
        self.win_type = win_type
        self.expect_dataset = expect_dataset

    def __iter__(self) -> Union[xr.DataArray, xr.Dataset]:
        """Compute rolling mean of PV power"""
        for xr_data in self.source_datapipe:
            if self.expect_dataset:
                data_to_resample = xr_data["power_w"]
            else:
                data_to_resample = xr_data

            resampled = data_to_resample.rolling(
                dim={"time_utc": self.window},
                min_periods=self.min_periods,
                center=self.center,
            ).mean()

            if self.expect_dataset:
                xr_data["power_w"] = resampled
                resampled = xr_data

            # Resampling removes the attributes, so put them back:
            for attr_name in ("t0_idx", "sample_period_duration"):
                resampled.attrs[attr_name] = xr_data.attrs[attr_name]

            yield resampled
