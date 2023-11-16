from datetime import timedelta

import numpy as np
import pandas as pd

from torch.utils.data.datapipes.iter import IterableWrapper
from ocf_datapipes.transform.xarray import GetContiguousT0TimePeriods, GetContiguousT0TimePeriodsNWP


def _remove_indexes(x, inds):
    xs = []
    i_last = -1
    for i in np.sort(inds):
        xs += [x[i_last + 1 : i]]
        i_last = i
    xs += [x[i_last + 1 :]]
    return pd.to_datetime(np.concatenate(xs))


def test_get_contiguous_time_periods(nwp_datapipe):
    # Create 5-minutely data timestamps
    freq = timedelta(minutes=5)
    history_duration = timedelta(minutes=60)
    forecast_duration = timedelta(minutes=15)

    datetimes = _remove_indexes(
        pd.date_range("2023-01-01 12:00", "2023-01-01 17:00", freq=freq),
        [5, 30],
    )

    # Create initial datapipe
    time_datapipe = IterableWrapper([pd.DataFrame(datetimes, columns=["time_utc"]).to_xarray()])

    history_duration = timedelta(minutes=60)

    contig_t0_datapipe = GetContiguousT0TimePeriods(
        time_datapipe,
        sample_period_duration=freq,
        history_duration=history_duration,
        forecast_duration=forecast_duration,
        time_dim="time_utc",
    )

    periods = next(iter(contig_t0_datapipe))

    expected_results = pd.DataFrame(
        {
            "start_dt": pd.to_datetime(
                [
                    "2023-01-01 13:30:00",
                    "2023-01-01 15:35:00",
                ]
            ),
            "end_dt": pd.to_datetime(
                [
                    "2023-01-01 14:10:00",
                    "2023-01-01 16:45:00",
                ]
            ),
        },
    )

    assert periods.equals(expected_results)


def test_get_contiguous_time_periods_nwp():
    # These are the expected results of the test
    expected_results = [
        pd.DataFrame(
            {
                "start_dt": pd.to_datetime(["2023-01-01 03:00:00", "2023-01-02 03:00:00"]),
                "end_dt": pd.to_datetime(["2023-01-01 21:00:00", "2023-01-03 06:00:00"]),
            },
        ),
        pd.DataFrame(
            {
                "start_dt": pd.to_datetime(
                    [
                        "2023-01-01 05:00:00",
                        "2023-01-02 05:00:00",
                        "2023-01-02 14:00:00",
                    ]
                ),
                "end_dt": pd.to_datetime(
                    [
                        "2023-01-01 21:00:00",
                        "2023-01-02 12:00:00",
                        "2023-01-03 06:00:00",
                    ]
                ),
            },
        ),
        pd.DataFrame(
            {
                "start_dt": pd.to_datetime(
                    [
                        "2023-01-01 05:00:00",
                        "2023-01-01 11:00:00",
                        "2023-01-02 05:00:00",
                        "2023-01-02 14:00:00",
                    ]
                ),
                "end_dt": pd.to_datetime(
                    [
                        "2023-01-01 09:00:00",
                        "2023-01-01 18:00:00",
                        "2023-01-02 09:00:00",
                        "2023-01-03 03:00:00",
                    ]
                ),
            },
        ),
        pd.DataFrame(
            {
                "start_dt": pd.to_datetime(
                    [
                        "2023-01-01 05:00:00",
                        "2023-01-01 11:00:00",
                        "2023-01-01 14:00:00",
                        "2023-01-02 05:00:00",
                        "2023-01-02 14:00:00",
                        "2023-01-02 17:00:00",
                        "2023-01-02 20:00:00",
                        "2023-01-02 23:00:00",
                    ]
                ),
                "end_dt": pd.to_datetime(
                    [
                        "2023-01-01 06:00:00",
                        "2023-01-01 12:00:00",
                        "2023-01-01 15:00:00",
                        "2023-01-02 06:00:00",
                        "2023-01-02 15:00:00",
                        "2023-01-02 18:00:00",
                        "2023-01-02 21:00:00",
                        "2023-01-03 00:00:00",
                    ]
                ),
            },
        ),
    ]

    # Create 3-hourly init times with a few time stamps missing
    freq = timedelta(minutes=180)

    datetimes = _remove_indexes(
        pd.date_range("2023-01-01 03:00", "2023-01-02 21:00", freq=freq),
        [1, 4, 5, 6, 7, 9, 10],
    )

    # Choose some history durations and max stalenesses
    history_durations_hr = [0, 2, 2, 2]
    max_stalenesses_hr = [9, 9, 6, 3]

    for i in range(len(expected_results)):
        history_duration = timedelta(hours=history_durations_hr[i])
        max_staleness = timedelta(hours=max_stalenesses_hr[i])

        # Create initial datapipe
        time_datapipe = IterableWrapper(
            [pd.DataFrame(datetimes, columns=["init_time_utc"]).to_xarray()]
        )

        time_periods = time_datapipe.get_contiguous_time_periods_nwp(
            history_duration=history_duration,
            max_staleness=max_staleness,
            time_dim="init_time_utc",
        )

        # Check if results are as expected
        results = next(iter(time_periods))
        assert results.equals(expected_results[i])
