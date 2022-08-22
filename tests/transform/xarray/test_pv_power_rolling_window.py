from ocf_datapipes.transform.xarray import PVPowerRollingWindow


def test_pv_power_rolling_window_passiv(passiv_dp):
    passiv_dp = PVPowerRollingWindow(passiv_dp, expect_dataset=False)
    data = next(iter(passiv_dp))
    assert data is not None
