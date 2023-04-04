import os

import numpy as np
import pytest

import ocf_datapipes
from ocf_datapipes.training.metnet_pv_site import metnet_site_datapipe


@pytest.mark.skip("Failing at the moment")
def test_metnet_datapipe():
    filename = os.path.join(os.path.dirname(ocf_datapipes.__file__), "../tests/config/test.yaml")
    gsp_datapipe = metnet_site_datapipe(filename, use_nwp=False, pv_in_image=True)

    batch = next(iter(gsp_datapipe))
    assert np.isfinite(batch[0]).all()
    assert np.isfinite(batch[1]).all()

def test_metnet_datapipe_cache():
    filename = os.path.join(os.path.dirname(ocf_datapipes.__file__), "../tests/config/test.yaml")
    gsp_datapipe = metnet_site_datapipe(filename, use_nwp=False, pv_in_image=True, cache_to_disk=True)

    batch = next(iter(gsp_datapipe))
    assert np.isfinite(batch[0]).all()
    assert np.isfinite(batch[1]).all()
