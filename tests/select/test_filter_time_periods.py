from datetime import datetime

import pandas as pd
from torch.utils.data.datapipes.datapipe import IterDataPipe

from ocf_datapipes.select import FilterTimePeriods


def test_select_time_slice_gsp(gsp_datapipe):
    start = pd.to_datetime(datetime(2020, 4, 1, 12))
    end = pd.to_datetime(datetime(2022, 4, 1, 16))

    class FakeTimePeriods(IterDataPipe):
        def __iter__(self):
            yield pd.DataFrame(columns=["start_dt", "end_dt"], data=[[start, end]])

    select_timer_periods_datapipe = FilterTimePeriods(gsp_datapipe, time_periods=FakeTimePeriods())

    data = next(iter(select_timer_periods_datapipe))

    assert (data.time_utc <= end).all()
    assert (data.time_utc >= start).all()
