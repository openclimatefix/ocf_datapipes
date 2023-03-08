import os

import numpy as np
import pytest

import ocf_datapipes
from ocf_datapipes.training.pseudo_irradience import pseudo_irradiance_datapipe


@pytest.mark.skip("Too big for CI at the moment")
def test_metnet_datapipe():
    filename = os.path.join(os.path.dirname(ocf_datapipes.__file__), "../tests/config/test.yaml")
    gsp_datapipe = pseudo_irradiance_datapipe(filename, use_nwp=False, size=32)

    batch = next(iter(gsp_datapipe))
    x = np.nan_to_num(batch[0])
    assert np.isfinite(x).all()
    assert not np.isnan(batch[1]).any()
    assert np.isfinite(batch[2]).all()
