from ocf_datapipes.transform.xarray import PVPowerRollingWindow, AddT0IdxAndSamplePeriodDuration
from datetime import timedelta

def test_pv_power_rolling_window_passiv(passiv_dp):
    passiv_dp = AddT0IdxAndSamplePeriodDuration(passiv_dp, history_duration=timedelta(minutes=60), sample_period_duration=timedelta(minutes=5))
    passiv_dp = PVPowerRollingWindow(passiv_dp, expect_dataset=False)
    data = next(iter(passiv_dp))
    assert data is not None
