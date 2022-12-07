from ocf_datapipes.select import DropNightPV


def test_drop_night_pv(simple_netcdf_datapipe):
    night_drop_pv = DropNightPV(simple_netcdf_datapipe)
    data = len(next(iter(night_drop_pv)))
    assert data is not None
