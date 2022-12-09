from ocf_datapipes.select import SelectSysWithoutOutputWholeday as Syswithoutpv
from ocf_datapipes.select import RemoveBadSystems
import logging

logger = logging.getLogger(__name__)


def test_sys_without_pv(simple_netcdf_datapipe):
    no_pv_wholeday = Syswithoutpv(simple_netcdf_datapipe)
    data = next(iter(no_pv_wholeday))
    key1 = sorted(data.keys())
    key2 = sorted(list({k2 for v in data.values() for k2 in v}))
    for x in key1:
        for y in key2:
            assert data[x][y] == "Active" or "Inactive"

def test_sys_without_passivpv(passiv_datapipe):
    no_pv_wholeday = RemoveBadSystems(passiv_datapipe)
    data = next(iter(no_pv_wholeday))
    assert len(list(data)) !=0