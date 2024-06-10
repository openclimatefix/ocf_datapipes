"""Selects time slice"""

import logging
from datetime import timedelta
from typing import Optional, Union

import numpy as np
import pandas as pd
import xarray as xr
from torch.utils.data import IterDataPipe, functional_datapipe

logger = logging.getLogger(__name__)


def fill_1d_bool_gaps(x, max_gap, fill_ends=False):
    """In a boolean array, fill consecutive False elements if their number is less than the gap_size

    Args:
        x: A 1-dimensional boolean array
        max_gap: integer of the maximum gap size which will be filled with True
        fill_ends: Whether to fill the ends as if there are True values on either side

    Returns:
        A 1-dimensional boolean array

    Examples:
        >>> x = np.array([0, 1, 0, 0, 1, 0, 1, 0])
        >>> fill_1d_bool_gaps(x, max_gap=2, fill_ends=False).astype(int)
        array([0, 1, 1, 1, 1, 1, 1, 0])

        >>> x = np.array([0, 1, 0, 0, 1, 0, 1, 0])
        >>> fill_1d_bool_gaps(x, max_gap=1, fill_ends=True).astype(int)
        array([1, 1, 0, 0, 1, 1, 1, 1])
    """
    if fill_ends:
        x_extended = np.concatenate([[True], x, [True]])
        return fill_1d_bool_gaps(x_extended, max_gap, fill_ends=False)[1:-1]

    should_fill = np.zeros(len(x), dtype=bool)

    i_start = None

    last_b = False
    for i, b in enumerate(x):
        if last_b and not b:
            i_start = i
        elif b and not last_b and i_start is not None:
            if i - i_start <= max_gap:
                should_fill[i_start:i] = True
            i_start = None
        last_b = b

    return np.logical_or(should_fill, x)


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
        max_steps_gap: Optional[int] = 0,
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
                in the returned xarray object tyo give the expected shape. Else the default xarray
                slicing behaviour is used and timestamps may be missing. When filled, the values are
                linearly interpolated up to a gap size of `max_steps_gap` steps. If outside this
                range, the values are set to NaN.
            max_steps_gap (optional): The number of consecutive missing time steps which will be
                filled via linear interpolation. If set to zero, no interpolation is used and all
                missing timesteps will be NaN.
        """
        self.source_datapipe = source_datapipe
        self.t0_datapipe = t0_datapipe
        self.fill_selection = fill_selection
        self.max_steps_gap = max_steps_gap

        used_duration = history_duration is not None and forecast_duration is not None
        used_intervals = interval_start is not None and interval_end is not None
        assert used_duration ^ used_intervals, "Either durations, or intervals must be supplied"
        assert max_steps_gap >= 0, "max_steps_gap must be >= 0 "

        if used_duration:
            self.interval_start = -np.timedelta64(history_duration)
            self.interval_end = np.timedelta64(forecast_duration)
        elif used_intervals:
            self.interval_start = np.timedelta64(interval_start)
            self.interval_end = np.timedelta64(interval_end)

        self.sample_period_duration = sample_period_duration

        if self.fill_selection and max_steps_gap == 0:
            self._sel = self._sel_fillnan
        elif self.fill_selection and max_steps_gap > 0:
            self._sel = self._sel_fillinterp
        else:
            self._sel = self._sel_default

    def _sel_fillnan(self, xr_data, start_dt, end_dt):
        requested_times = pd.date_range(
            start_dt,
            end_dt,
            freq=self.sample_period_duration,
        )
        # Missing time indexes are returned with all NaN values
        return xr_data.reindex(time_utc=requested_times)

    def _sel_fillinterp(self, xr_data, start_dt, end_dt):
        dt_buffer = self.sample_period_duration * self.max_steps_gap

        # Initially select larger period so we can use it to interpolate to requested period
        # This slice also avoids us interpolating the whole dataset to get the requested times
        ds = xr_data.sel(time_utc=slice(start_dt - dt_buffer, end_dt + dt_buffer))

        # These are the times we will ultimately return
        requested_times = pd.date_range(
            start_dt,
            end_dt,
            freq=self.sample_period_duration,
        )

        # These are the times we use for interpolation to the requested_times
        buffer_requested_times = pd.date_range(
            start_dt - dt_buffer,
            end_dt + dt_buffer,
            freq=self.sample_period_duration,
        )

        # If all the requested times are present we avoid running interpolation
        if np.isin(requested_times, ds.time_utc).all():
            return ds.sel(time_utc=slice(start_dt, end_dt))

        # If less than 2 of the buffer requested times are present we cannot infill
        elif np.isin(buffer_requested_times, ds.time_utc).sum() < 2:
            logger.warning("Cannot run interpolate infilling with less than 2 time steps available")
            return self._sel_fillnan(xr_data, start_dt, end_dt)

        logger.info("Some requested times are missing - running interpolation")
        # Find the timestamps which are within max gap size
        mask = np.isin(buffer_requested_times, ds.time_utc)
        valid_fill_times = fill_1d_bool_gaps(mask, self.max_steps_gap, fill_ends=False)

        # Run the interpolation and filter to requested times
        ds_interp = ds.interp(time_utc=buffer_requested_times, method="linear", assume_sorted=True)

        # Mask the timestamps outside the max gap size
        valid_fill_times_xr = xr.zeros_like(ds_interp.time_utc, dtype=bool)
        valid_fill_times_xr.values[:] = valid_fill_times

        valid_requested_times = valid_fill_times_xr.sel(time_utc=slice(start_dt, end_dt))
        if not valid_requested_times.all():
            not_infilled_times = valid_requested_times.where(~valid_requested_times, drop=True)
            logger.warning(
                "After interpolation the following requested times are still missing:"
                f"{not_infilled_times.time_utc.values}"
            )

        ds_out = ds_interp.where(valid_fill_times_xr)

        # Filter to selected times
        ds_out = ds_out.sel(time_utc=slice(start_dt, end_dt))
        return ds_out

    def _sel_default(self, xr_data, start_dt, end_dt):
        return xr_data.sel(time_utc=slice(start_dt, end_dt))

    def __iter__(self) -> Union[xr.DataArray, xr.Dataset]:
        for t0, xr_data in self.t0_datapipe.zip(self.source_datapipe):
            t0_datetime_utc = pd.Timestamp(t0)
            start_dt = t0_datetime_utc + self.interval_start
            end_dt = t0_datetime_utc + self.interval_end

            start_dt = start_dt.ceil(self.sample_period_duration)
            end_dt = end_dt.ceil(self.sample_period_duration)

            yield self._sel(xr_data, start_dt, end_dt)
