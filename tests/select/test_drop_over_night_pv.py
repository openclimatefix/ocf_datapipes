from ocf_datapipes.select import DropNightPV


def test_drop_night_pv(simple_netcdf_datapipe):
    night_drop_pv = DropNightPV(simple_netcdf_datapipe)
    data = next(iter(night_drop_pv))
    assert "day" or "night" in data.coords["daynight_status"].values
