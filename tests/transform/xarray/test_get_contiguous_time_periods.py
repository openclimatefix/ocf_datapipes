from datetime import timedelta

import numpy as np
import pandas as pd

from torchdata.datapipes.iter import IterableWrapper
from ocf_datapipes.transform.xarray import (
    GetContiguousT0TimePeriods, GetContiguousT0TimePeriodsNWP
)


def get_contiguous_time_periods_nwp(nwp_datapipe):
    nwp_datapipe = GetContiguousT0TimePeriods(
        nwp_datapipe,
        sample_period_duration=timedelta(hours=3),
        history_duration=timedelta(minutes=60),
        forecast_duration=timedelta(minutes=180),
        time_dim="init_time_utc",
    )

    batch = next(iter(nwp_datapipe))


def test_get_contiguous_time_periods():
    
    # These are the expected results of the test
    expected_results = [
        pd.DataFrame(
            {
                "start_dt":pd.to_datetime(["2023-01-01 03:00:00", "2023-01-02 03:00:00"]), 
                "end_dt":pd.to_datetime(["2023-01-01 21:00:00", "2023-01-03 06:00:00"])
            }, 
        ),
        pd.DataFrame(
            {
                "start_dt":pd.to_datetime(["2023-01-01 06:00:00", "2023-01-02 06:00:00"]), 
                "end_dt":pd.to_datetime(["2023-01-01 21:00:00", "2023-01-03 06:00:00"])
            }, 
        ),
        pd.DataFrame(
            {
                "start_dt":pd.to_datetime([
                    "2023-01-01 06:00:00", "2023-01-02 06:00:00", "2023-01-02 15:00:00",
                ]), 
                "end_dt":pd.to_datetime([
                    "2023-01-01 18:00:00", "2023-01-02 09:00:00", "2023-01-03 03:00:00",
                ])
            }, 
        ),
        pd.DataFrame(
            {
                "start_dt":pd.to_datetime([
                    "2023-01-01 06:00:00", "2023-01-01 12:00:00",
                    "2023-01-02 06:00:00", "2023-01-02 15:00:00",
                ]), 
                "end_dt":pd.to_datetime([
                    "2023-01-01 06:00:00", "2023-01-01 15:00:00",
                    "2023-01-02 06:00:00", "2023-01-03 00:00:00",
                ])
            }, 
        ),
    ]

    def _remove_indexes(x, inds):
        xs = []
        i_last = -1
        for i in np.sort(inds):
            xs += [x[i_last+1:i]]
            i_last = i
        xs += [x[i_last+1:]]
        return pd.to_datetime(np.concatenate(xs))

    # Create 3-hourly init times with a few time stamps missing
    freq = timedelta(minutes=180)

    datetimes = _remove_indexes(
        pd.date_range("2023-01-01 03:00", "2023-01-02 21:00", freq=freq),
        [1,4,5,6,7,9,10],
    )
    
    # Choose some history durations and max stalenesses
    history_durations_hr = [0,3,3,3]
    max_stalenesses_hr = [9,9,6,3]


    for i in range(len(expected_results)):
        history_duration = timedelta(hours=history_durations_hr[i])
        max_staleness = timedelta(hours=max_stalenesses_hr[i])
        
        # Create initial datapipe
        datapipe_copy = IterableWrapper(
            [pd.DataFrame(datetimes, columns=["init_time_utc"]).to_xarray()]
        )

        time_periods = datapipe_copy.get_contiguous_time_periods_nwp(
            history_duration=history_duration,
            max_staleness=max_staleness,
            time_dim="init_time_utc",
        )
        
        # Check if results are as expected
        results = next(iter(time_periods))
        assert results.equals(expected_results[i])

    