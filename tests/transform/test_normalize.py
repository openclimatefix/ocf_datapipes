from ocf_datapipes.load import OpenNWP, OpenSatellite, OpenTopography, OpenGSP, OpenPVFromNetCDF
from ocf_datapipes.transform.xarray import Normalize


def test_normalize_sat():
    sat_dp = OpenSatellite(zarr_path="tests/data/hrv_sat_data.zarr")
    sat_dp = Normalize(sat_dp, mean=[0.5], std=[0.5])
    data = next(iter(sat_dp))
    assert data is not None

def test_normalize_nwp():
    pass

def test_normalize_topo():
    pass

def test_normalize_gsp():
    pass

def test_normalize_pv():
    pass
