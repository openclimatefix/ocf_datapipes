from ocf_datapipes.load import OpenGSP, OpenNWP, OpenPVFromNetCDF, OpenSatellite, OpenTopography
from ocf_datapipes.transform.xarray import Normalize


def test_normalize_sat(sat_dp):
    sat_dp = Normalize(sat_dp, mean=[0.5], std=[0.5])
    data = next(iter(sat_dp))
    assert data is not None


def test_normalize_nwp(nwp_dp):
    pass


def test_normalize_topo(topo_dp):
    pass


def test_normalize_gsp():
    pass


def test_normalize_passiv(passiv_dp):
    pass


def test_normalize_pvoutput(pvoutput_dp):
    pass
