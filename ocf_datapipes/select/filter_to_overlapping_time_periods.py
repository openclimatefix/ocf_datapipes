"""Select overlapping time slices for training"""

import logging
from typing import Iterable, Optional

import pandas as pd
from torch.utils.data import IterDataPipe, functional_datapipe

logger = logging.getLogger(__name__)


@functional_datapipe("filter_to_overlapping_time_periods")
class FilterToOverlappingTimePeriodsIterDataPipe(IterDataPipe):
    """Filter to overlapping time periods"""

    def __init__(
        self,
        source_datapipe: IterDataPipe,
        secondary_datapipes: Iterable[IterDataPipe],
        location_datapipe: Optional[IterDataPipe] = None,
    ):
        """Filter to overlapping time periods

        Source DataPipes are from the contiguous_time_period

        Args:
            source_datapipe: First Datapipe to compute the intersection of
            secondary_datapipes: More Datapipe to compute the intersection of
            location_datapipe: location fo data pipe. This allows quick loading of
                time periods if present
        """
        super().__init__()
        self.source_datapipe = source_datapipe
        self.secondary_datapipes = secondary_datapipes
        self.location_datapipe = location_datapipe
        self.time_periods_id = {}

    def __iter__(self) -> pd.DataFrame:
        if self.location_datapipe is None:
            for set_of_pd_datas in self.source_datapipe.zip_ocf(*self.secondary_datapipes):
                time_periods = intersection_of_multiple_dataframes_of_periods(list(set_of_pd_datas))

                logger.debug(f"Found {len(time_periods)} time periods")
                assert len(time_periods) > 0, "\n".join([str(df) for df in set_of_pd_datas])

                yield time_periods
        else:
            for set_of_pd_datas in self.source_datapipe.zip_ocf(
                *self.secondary_datapipes, self.location_datapipe
            ):
                location = set_of_pd_datas[-1]
                set_of_pd_datas = set_of_pd_datas[:-1]

                id = int(location.id)

                if id in self.time_periods_id.keys():
                    logger.debug(f"Time periods for id {id} from store")
                    logger.debug(len(self.time_periods_id[id]))
                    time_periods = self.time_periods_id[id]
                else:
                    logger.debug(f"Time periods for id {id} not in store")
                    time_periods = intersection_of_multiple_dataframes_of_periods(
                        list(set_of_pd_datas)
                    )

                    logger.debug(f"Found {len(time_periods)} time periods")
                    assert len(time_periods) > 0, "\n".join([str(df) for df in set_of_pd_datas])

                    self.time_periods_id[id] = time_periods
                    logger.debug(len(time_periods))

                yield time_periods


def intersection_of_multiple_dataframes_of_periods(
    time_periods: list[pd.DataFrame],
) -> pd.DataFrame:
    """Find the intersection of a list of time periods.

    See the docstring of intersection_of_2_dataframes_of_periods() for more details.
    """
    assert len(time_periods) > 0
    if len(time_periods) == 1:
        return time_periods[0]
    intersection = intersection_of_2_dataframes_of_periods(time_periods[0], time_periods[1])
    for time_period in time_periods[2:]:
        intersection = intersection_of_2_dataframes_of_periods(intersection, time_period)
    return intersection


def intersection_of_2_dataframes_of_periods(a: pd.DataFrame, b: pd.DataFrame) -> pd.DataFrame:
    """Find the intersection of two pd.DataFrames of time periods.

    Each row of each pd.DataFrame represents a single time period.  Each pd.DataFrame has
    two columns: `start_dt` and `end_dt` (where 'dt' is short for 'datetime').

    A typical use-case is that each pd.DataFrame represents all the time periods where
    a `DataSource` has contiguous, valid data.

    Here's a graphical example of two pd.DataFrames of time periods and their intersection:

                 ----------------------> TIME ->---------------------
               a: |-----|   |----|     |----------|     |-----------|
               b:    |--------|                       |----|    |---|
    intersection:    |--|   |-|                         |--|    |---|

    Args:
        a: pd.DataFrame where each row represents a time period.  The pd.DataFrame has
        two columns: start_dt and end_dt.
        b: pd.DataFrame where each row represents a time period.  The pd.DataFrame has
        two columns: start_dt and end_dt.

    Returns:
        Sorted list of intersecting time periods represented as a pd.DataFrame with two columns:
        start_dt and end_dt.
    """
    if a.empty or b.empty:
        return pd.DataFrame(columns=["start_dt", "end_dt"])

    all_intersecting_periods = []
    for a_period in a.itertuples():
        # Five ways in which two periods may overlap:
        # a: |----| or |---|   or  |---| or   |--|   or |-|
        # b:  |--|       |---|   |---|      |------|    |-|
        # In all five, `a` must always start before `b` ends,
        # and `a` must always end after `b` starts:
        overlapping_periods = b[(a_period.start_dt < b.end_dt) & (a_period.end_dt > b.start_dt)]

        # There are two ways in which two periods may *not* overlap:
        # a: |---|        or        |---|
        # b:       |---|      |---|
        # `overlapping` will not include periods which do *not* overlap.

        # Now find the intersection of each period in `overlapping_periods` with
        # the period from `a` that starts at `a_start_dt` and ends at `a_end_dt`.
        # We do this by clipping each row of `overlapping_periods`
        # to start no earlier than `a_start_dt`, and end no later than `a_end_dt`.

        # First, make a copy, so we don't clip the underlying data in `b`.
        intersection = overlapping_periods.copy()
        intersection["start_dt"] = intersection.start_dt.clip(lower=a_period.start_dt)
        intersection["end_dt"] = intersection.end_dt.clip(upper=a_period.end_dt)

        all_intersecting_periods.append(intersection)

    all_intersecting_periods = pd.concat(all_intersecting_periods)
    return all_intersecting_periods.sort_values(by="start_dt").reset_index(drop=True)
