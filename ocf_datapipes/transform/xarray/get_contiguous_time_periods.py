"""Get contiguous time periods for training"""
from datetime import timedelta

import numpy as np
import pandas as pd
from torchdata.datapipes import functional_datapipe
from torchdata.datapipes.iter import IterDataPipe


@functional_datapipe("add_contiguous_time_periods")
class GetContiguousT0TimePeriodsIterDataPipe(IterDataPipe):
    """Get contiguous time periods for training"""

    def __init__(
        self,
        source_datapipe: IterDataPipe,
        history_duration: timedelta,
        forecast_duration: timedelta,
        sample_period_duration: timedelta,
    ):
        """
        Get contiguous time periods for use in determing t0 times for training

        Args:
            source_datapipe: Datapipe emitting a Xarray dataset
            history_duration: Amount of time for the history of an example
            forecast_duration: Amount of time for the forecast of an example
            sample_period_duration: The sampling period of the data source
        """
        self.source_datapipe = source_datapipe
        self.history_duration = history_duration
        self.forecast_duration = forecast_duration
        self.total_duration = history_duration + forecast_duration
        self.sample_period_duration = sample_period_duration

    def __iter__(self) -> pd.DataFrame:
        """Calculate contiguous time periods and return a dataframe containing them"""
        for xr_data in self.source_datapipe:
            contiguous_time_periods = get_contiguous_time_periods(
                datetimes=pd.DatetimeIndex(xr_data["time_utc"]),
                min_seq_length=int(self.total_duration / self.sample_period_duration) + 1,
                max_gap_duration=self.sample_period_duration,
            )
            contiguous_time_periods = get_contiguous_t0_time_periods(
                contiguous_time_periods=contiguous_time_periods,
                history_duration=self.history_duration,
                forecast_duration=self.forecast_duration,
            )
            yield contiguous_time_periods


def get_contiguous_t0_time_periods(
    contiguous_time_periods: pd.DataFrame, history_duration: timedelta, forecast_duration: timedelta
) -> pd.DataFrame:
    """Get all time periods which contain valid t0 datetimes.

    `t0` is the datetime of the most recent observation.

    Returns:
      pd.DataFrame where each row represents a single time period.  The pd.DataFrame
      has two columns: `start_dt` and `end_dt` (where 'dt' is short for 'datetime').
    """
    contiguous_time_periods["start_dt"] += history_duration
    contiguous_time_periods["end_dt"] -= forecast_duration
    assert (contiguous_time_periods["start_dt"] < contiguous_time_periods["end_dt"]).all()
    return contiguous_time_periods


def get_contiguous_time_periods(
    datetimes: pd.DatetimeIndex,
    min_seq_length: int,
    max_gap_duration: timedelta,
) -> pd.DataFrame:
    """Return a pd.DataFrame where each row records the boundary of a contiguous time period.

    Args:
      datetimes: pd.DatetimeIndex. Must be sorted.
      min_seq_length: Sequences of min_seq_length or shorter will be discarded.  Typically, this
        would be set to the `total_seq_length` of each machine learning example.
      max_gap_duration: If any pair of consecutive `datetimes` is more than `max_gap_duration`
        apart, then this pair of `datetimes` will be considered a "gap" between two contiguous
        sequences. Typically, `max_gap_duration` would be set to the sample period of
        the timeseries.

    Returns:
      pd.DataFrame where each row represents a single time period.  The pd.DataFrame
          has two columns: `start_dt` and `end_dt` (where 'dt' is short for 'datetime').
    """
    # Sanity checks.
    assert len(datetimes) > 0
    assert min_seq_length > 1
    assert datetimes.is_monotonic_increasing
    assert datetimes.is_unique

    # Find indices of gaps larger than max_gap:
    gap_mask = pd.TimedeltaIndex(np.diff(datetimes)) > max_gap_duration
    gap_indices = np.argwhere(gap_mask)[:, 0]

    # gap_indicies are the indices into dt_index for the timestep immediately before the gap.
    # e.g. if the datetimes at 12:00, 12:05, 18:00, 18:05 then gap_indicies will be [1].
    # So we add 1 to gap_indices to get segment_boundaries, an index into dt_index
    # which identifies the _start_ of each segment.
    segment_boundaries = gap_indices + 1

    # Capture the last segment of dt_index.
    segment_boundaries = np.concatenate((segment_boundaries, [len(datetimes)]))

    periods: list[dict[str, pd.Timestamp]] = []
    start_i = 0
    for next_start_i in segment_boundaries:
        n_timesteps = next_start_i - start_i
        if n_timesteps > min_seq_length:
            end_i = next_start_i - 1
            period = {"start_dt": datetimes[start_i], "end_dt": datetimes[end_i]}
            periods.append(period)
        start_i = next_start_i

    assert len(periods) > 0

    return pd.DataFrame(periods)
