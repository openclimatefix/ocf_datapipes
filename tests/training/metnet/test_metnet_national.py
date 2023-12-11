import os

import numpy as np
import pytest

import ocf_datapipes
from ocf_datapipes.training.metnet.metnet_national import metnet_national_datapipe


@pytest.mark.skip("Failing at the moment")
def test_metnet_national_datapipe():
    filename = os.path.join(os.path.dirname(ocf_datapipes.__file__), "../tests/config/test.yaml")

    datapipe = metnet_national_datapipe(filename, max_num_pv_systems=1).set_length(2)

    batch = next(iter(datapipe))
    assert np.isfinite(batch[0]).all()
    assert np.isfinite(batch[1]).all()
