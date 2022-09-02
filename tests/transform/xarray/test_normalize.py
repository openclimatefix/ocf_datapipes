import numpy as np

from ocf_datapipes.transform.xarray import Normalize
from ocf_datapipes.utils.consts import NWP_MEAN, NWP_STD, SAT_MEAN, SAT_STD


def test_normalize_sat(sat_datapipe):
    sat_datapipe = Normalize(sat_datapipe, mean=SAT_MEAN["HRV"], std=SAT_STD["HRV"])
    data = next(iter(sat_datapipe))
    assert np.all(data <= 1.0)
    assert np.all(data >= -1.0)


def test_normalize_nwp(nwp_datapipe):
    nwp_datapipe = Normalize(nwp_datapipe, mean=NWP_MEAN, std=NWP_STD)
    data = next(iter(nwp_datapipe))
    assert np.all(data.values <= 1.0)
    # TODO Check why this normalization is so negative
    assert np.all(data.values >= -60.0)


def test_normalize_topo(topo_datapipe):
    topo_datapipe = topo_datapipe.reproject_topography().normalize(
        calculate_mean_std_from_example=True
    )
    data = next(iter(topo_datapipe))
    assert np.all(data >= -1.0)
    # TODO Check why this normalization is ends up with a max of 11ish instead
    assert np.all(data <= 11.0)


def test_normalize_gsp(gsp_datapipe):
    gsp_datapipe = gsp_datapipe.normalize(normalize_fn=lambda x: x / x.capacity_megawatt_power)
    data = next(iter(gsp_datapipe))
    assert np.min(data) >= 0.0
    assert np.max(data) <= 1.0


def test_normalize_passiv(passiv_datapipe):
    passiv_datapipe = passiv_datapipe.normalize(normalize_fn=lambda x: x / x.capacity_watt_power)
    data = next(iter(passiv_datapipe))
    assert np.min(data) >= 0.0
    assert np.max(data) <= 1.0


def test_normalize_pvoutput(pvoutput_datapipe):
    pvoutput_datapipe = pvoutput_datapipe.normalize(
        normalize_fn=lambda x: x / x.capacity_watt_power
    )
    data = next(iter(pvoutput_datapipe))
    assert np.min(data) >= 0.0
    assert np.max(data) <= 1.0
