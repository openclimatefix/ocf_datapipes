import numpy as np

from ocf_datapipes.transform.xarray import Normalize
from ocf_datapipes.utils.consts import NWP_MEANS, NWP_STDS, RSS_MEAN, RSS_STD
import pytest


def test_normalize_sat(sat_datapipe):
    normed_sat_datapipe = Normalize(
        sat_datapipe,
        # HRV dosn't have channel dimension
        mean=RSS_MEAN.sel(channel="HRV"),
        std=RSS_STD.sel(channel="HRV"),
    )
    data = next(iter(normed_sat_datapipe))
    assert np.all(data <= 10.0)
    assert np.all(data >= -10.0)


def test_normalize_nwp(nwp_datapipe):
    normed_nwp_datapipe = Normalize(nwp_datapipe, mean=NWP_MEANS["ukv"], std=NWP_STDS["ukv"])
    data = next(iter(normed_nwp_datapipe))
    assert np.all(data.values <= 10.0)
    assert np.all(data.values >= -10.0)


def test_normalize_topo(topo_datapipe):
    normed_topo_datapipe = topo_datapipe.reproject_topography().normalize(
        calculate_mean_std_from_example=True
    )
    data = next(iter(normed_topo_datapipe))
    assert data.mean().compute() == pytest.approx(0, abs=0.001)
    assert data.std().compute() == pytest.approx(1, abs=0.001)


def test_normalize_gsp(gsp_datapipe):
    normed_gsp_datapipe = gsp_datapipe.normalize(
        normalize_fn=lambda x: x / x.effective_capacity_mwp
    )
    data = next(iter(normed_gsp_datapipe))
    assert np.min(data) >= 0.0

    # Some GSPs are noisey and seem to have values above 1
    assert np.max(data) <= 1.5


def test_normalize_passiv(passiv_datapipe):
    normed_passiv_datapipe = passiv_datapipe.normalize(
        normalize_fn=lambda x: x / x.observed_capacity_wp
    )
    data = next(iter(normed_passiv_datapipe))
    assert np.min(data) >= 0.0
    assert np.max(data) <= 1.0


def test_normalize_pvoutput(pvoutput_datapipe):
    normed_pvoutput_datapipe = pvoutput_datapipe.normalize(
        normalize_fn=lambda x: x / x.observed_capacity_wp
    )
    data = next(iter(normed_pvoutput_datapipe))
    assert np.min(data) >= 0.0
    assert np.max(data) <= 1.0
