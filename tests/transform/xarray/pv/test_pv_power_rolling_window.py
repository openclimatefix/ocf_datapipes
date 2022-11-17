from datetime import timedelta

from ocf_datapipes.transform.xarray import AddT0IdxAndSamplePeriodDuration, PVPowerRollingWindow


def test_pv_power_rolling_window_passiv(passiv_datapipe):
    passiv_datapipe = AddT0IdxAndSamplePeriodDuration(
        passiv_datapipe,
        history_duration=timedelta(minutes=60),
        sample_period_duration=timedelta(minutes=5),
    )
    data_before = next(iter(passiv_datapipe))
    passiv_datapipe = PVPowerRollingWindow(passiv_datapipe, expect_dataset=False)
    data = next(iter(passiv_datapipe))
    assert len(data.time_utc) == len(data_before.time_utc)
