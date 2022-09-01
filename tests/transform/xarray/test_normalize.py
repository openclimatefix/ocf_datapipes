import numpy as np

from ocf_datapipes.transform.xarray import Normalize
from ocf_datapipes.utils.consts import NWP_MEAN, NWP_STD, SAT_MEAN, SAT_STD


def test_normalize_sat(sat_dp):
    sat_dp = Normalize(sat_dp, mean=SAT_MEAN["HRV"], std=SAT_STD["HRV"])
    data = next(iter(sat_dp))
    assert np.all(data <= 1.0)
    assert np.all(data >= -1.0)


def test_normalize_nwp(nwp_dp):
    nwp_dp = Normalize(nwp_dp, mean=NWP_MEAN, std=NWP_STD)
    data = next(iter(nwp_dp))
    assert np.all(data.values <= 1.0)
    # TODO Check why this normalization is so negative
    assert np.all(data.values >= -60.0)


def test_normalize_topo(topo_dp):
    topo_dp = topo_dp.reproject_topography().normalize(calculate_mean_std_from_example=True)
    data = next(iter(topo_dp))
    assert np.all(data >= -1.0)
    # TODO Check why this normalization is ends up with a max of 11ish instead
    assert np.all(data <= 11.0)


def test_normalize_gsp(gsp_dp):
    gsp_dp = gsp_dp.normalize(normalize_fn=lambda x: x / x.capacity_mwp)
    data = next(iter(gsp_dp))
    assert np.min(data) >= 0.0
    assert np.max(data) <= 1.0


def test_normalize_passiv(passiv_dp):
    passiv_dp = passiv_dp.normalize(normalize_fn=lambda x: x / x.capacity_wp)
    data = next(iter(passiv_dp))
    assert np.min(data) >= 0.0
    assert np.max(data) <= 1.0


def test_normalize_pvoutput(pvoutput_dp):
    pvoutput_dp = pvoutput_dp.normalize(normalize_fn=lambda x: x / x.capacity_wp)
    data = next(iter(pvoutput_dp))
    assert np.min(data) >= 0.0
    assert np.max(data) <= 1.0
