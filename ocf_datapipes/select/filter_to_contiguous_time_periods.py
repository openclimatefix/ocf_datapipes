"""Get contiguous time periods for training"""
import logging
from datetime import timedelta

import numpy as np
import pandas as pd
from torch.utils.data import IterDataPipe, functional_datapipe

logger = logging.getLogger(__name__)


@functional_datapipe("get_contiguous_time_periods")
class GetContiguousT0TimePeriodsIterDataPipe(IterDataPipe):
    """Get contiguous time periods for training"""

    def __init__(
        self,
        source_datapipe: IterDataPipe,
        history_duration: timedelta,
        forecast_duration: timedelta,
        sample_period_duration: timedelta,
        max_t0_offset: timedelta = timedelta(minutes=0),
        time_dim: str = "time_utc",
    ):
        """
        Get contiguous time periods for use in determing t0 times for training

        Args:
            source_datapipe: Datapipe emitting a Xarray dataset
            history_duration: Amount of time for the history of an example
            forecast_duration: Amount of time for the forecast of an example
            sample_period_duration: The sampling period of the data source
            max_t0_offset: Max t0 offset for the data source, add as buffer to total duration
            time_dim: time dimensions for which to find the contiguous time periods
        """
        self.source_datapipe = source_datapipe
        self.history_duration = history_duration
        self.forecast_duration = forecast_duration
        self.total_duration = history_duration + forecast_duration + max_t0_offset
        self.sample_period_duration = sample_period_duration
        self.max_t0_offset = max_t0_offset
        self.time_dim = time_dim

    def __iter__(self) -> pd.DataFrame:
        """Calculate contiguous time periods and return a dataframe containing them"""
        for xr_data in self.source_datapipe:
            logger.debug("Getting contiguous time periods")
            contiguous_time_periods = get_contiguous_time_periods(
                datetimes=pd.DatetimeIndex(xr_data[self.time_dim]),
                min_seq_length=int(self.total_duration / self.sample_period_duration) + 1,
                max_gap_duration=self.sample_period_duration,
            )
            logger.debug("Getting contiguous t0 time periods")
            contiguous_time_periods = get_contiguous_t0_time_periods(
                contiguous_time_periods=contiguous_time_periods,
                history_duration=self.history_duration,
                forecast_duration=self.forecast_duration,
            )
            logger.debug("Get contiguous time periods:done")
            yield contiguous_time_periods


@functional_datapipe("get_contiguous_time_periods_nwp")
class GetContiguousT0TimePeriodsNWPIterDataPipe(IterDataPipe):
    """Get contiguous NWP time periods for training"""

    def __init__(
        self,
        source_datapipe: IterDataPipe,
        history_duration: timedelta,
        max_staleness: timedelta = timedelta(minutes=0),
        max_dropout: timedelta = timedelta(minutes=0),
        time_dim: str = "init_time_utc",
    ):
        """
        Get contiguous time periods for use in determing t0 times for training

        Args:
            source_datapipe: Datapipe emitting a Xarray dataset
            history_duration: Length of the historical slice used for a sample
            max_staleness: Up to how long after an NWP forecast init_time are we willing to use the
                forecast. Each init time will only be used up to this t0 time regardless of the
                forecast valid time.
            max_dropout: What is the maximum amount of dropout that will be used. This must be <=
                max_staleness.
            time_dim: time dimensions for which to find the contiguous time periods
        """
        self.source_datapipe = source_datapipe
        self.history_duration = history_duration
        self.max_staleness = max_staleness
        self.max_dropout = max_dropout
        self.time_dim = time_dim

    def __iter__(self) -> pd.DataFrame:
        """Calculate contiguous time periods and return a dataframe containing them"""
        for xr_data in self.source_datapipe:
            logger.debug("Getting contiguous NWP t0 time periods")
            contiguous_time_periods = get_contiguous_t0_periods_nwp(
                datetimes=pd.DatetimeIndex(xr_data[self.time_dim]),
                history_duration=self.history_duration,
                max_staleness=self.max_staleness,
                max_dropout=self.max_dropout,
            )
            yield contiguous_time_periods


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

    assert len(periods) > 0, (
        f"Did not find an periods from {datetimes}. " f"{min_seq_length=} {max_gap_duration=}"
    )

    return pd.DataFrame(periods)


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


def get_contiguous_t0_periods_nwp(
    datetimes: pd.DatetimeIndex,
    history_duration: timedelta,
    max_staleness: timedelta,
    max_dropout: timedelta = timedelta(0),
) -> pd.DataFrame:
    """Get all time periods from the NWP init times which are valid as t0 datetimes.

    Args:
        datetimes: Sorted pd.DatetimeIndex
        history_duration: Length of the historical slice used for a sample
        max_staleness: Up to how long after an NWP forecast init_time are we willing to use the
            forecast. Each init time will only be used up to this t0 time regardless of the forecast
            valid time.
        max_dropout: What is the maximum amount of dropout that will be used. This must be <=
            max_staleness.

    Returns:
        pd.DataFrame where each row represents a single time period.  The pd.DataFrame
        has two columns: `start_dt` and `end_dt` (where 'dt' is short for 'datetime').
    """
    # Sanity checks.
    assert len(datetimes) > 0
    assert datetimes.is_monotonic_increasing
    assert datetimes.is_unique
    assert history_duration >= timedelta(0)
    assert max_staleness >= timedelta(0)
    assert max_dropout <= max_staleness

    hist_drop_buffer = max(history_duration, max_dropout)

    # Store contiguous periods
    contiguous_periods = []

    # Start first period allowing for history slice and max dropout
    start_this_period = datetimes[0] + hist_drop_buffer

    # The first forecast is valid up to the max staleness
    end_this_period = datetimes[0] + max_staleness

    for dt_init in datetimes[1:]:
        # If the previous init time becomes stale before the next init becomes valid whilst also
        # considering dropout and the need for a historic period - then the contiguous period breaks
        if end_this_period < dt_init + hist_drop_buffer:
            contiguous_periods += [[start_this_period, end_this_period]]

            # And start a new period
            start_this_period = dt_init + hist_drop_buffer
        end_this_period = dt_init + max_staleness

    contiguous_periods += [[start_this_period, end_this_period]]

    return pd.DataFrame(contiguous_periods, columns=["start_dt", "end_dt"])
