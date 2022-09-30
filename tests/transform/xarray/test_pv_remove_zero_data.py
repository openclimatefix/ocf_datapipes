from datetime import timedelta

import numpy as np

from ocf_datapipes.transform.xarray import AddT0IdxAndSamplePeriodDuration, PVPowerRemoveZeroData


def test_pv_power_remove_data(passiv_datapipe):
    passiv_datapipe = AddT0IdxAndSamplePeriodDuration(
        passiv_datapipe,
        history_duration=timedelta(minutes=60),
        sample_period_duration=timedelta(minutes=5),
    )
    data_before = next(iter(passiv_datapipe))

    print(data_before)

    # set some valyes to zeros
    # data_before.values[0] = 0

    passiv_datapipe = PVPowerRemoveZeroData(passiv_datapipe, window=timedelta(minutes=60))
    data = next(iter(passiv_datapipe))

    assert np.isnan(data.values[0, 0])
    assert len(data.time_utc) == len(data_before.time_utc)
