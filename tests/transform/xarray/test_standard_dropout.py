from datetime import timedelta
from torch.utils.data.datapipes.iter import IterableWrapper
import pandas as pd
import numpy as np

from ocf_datapipes.transform.xarray import SelectDropoutTime, ApplyDropoutTime


def test_select_dropout_time(sat_datapipe):
    data = next(iter(sat_datapipe))

    t0_times = pd.to_datetime(data.time_utc.values)

    t0_datapipe = IterableWrapper(t0_times)

    # All times delayed by 5 minutes
    dropout_time_datapipe = SelectDropoutTime(
        source_datapipe=t0_datapipe,
        dropout_timedeltas=[timedelta(minutes=-5)],
        dropout_frac=1,
    )

    dropout_times = [*dropout_time_datapipe]

    assert (np.array(dropout_times) == t0_times - timedelta(minutes=5)).all()


def test_apply_dropout_time(sat_datapipe):
    data = next(iter(sat_datapipe)).isel(x_geostationary=0, y_geostationary=0).compute()

    times = pd.to_datetime(data.time_utc.values[3:])

    dropout_time_datapipe = IterableWrapper(times - timedelta(minutes=5))

    sliced_sat_datapipe = IterableWrapper(
        [data.sel(time_utc=slice(t - timedelta(minutes=10), t)) for t in times]
    )

    dropout_out_sat = ApplyDropoutTime(
        source_datapipe=sliced_sat_datapipe,
        dropout_time_datapipe=dropout_time_datapipe,
    )

    # Only the last element of the slice should be nan
    for sat_im in dropout_out_sat:
        assert (np.isnan(sat_im.values).squeeze() == np.array([False, False, True])).all()
