from ocf_datapipes.select import SelectSysWithoutOutputWholeday as syswithoutpv

def test_sys_without_pv(simple_netcdf_datapipe):
    no_pv_wholeday = syswithoutpv(simple_netcdf_datapipe)
    data = next(iter(no_pv_wholeday))
    key1 = sorted(data.keys())
    key2 = sorted(list({k2 for v in data.values() for k2 in v}))
    for x in key1:
        for y in key2:
            assert data[x][y] == 'Active' or 'Inactive'

