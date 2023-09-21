import os

import numpy as np
import pytest
import torch

import ocf_datapipes
from ocf_datapipes.training.pseudo_irradience import pseudo_irradiance_datapipe


@pytest.mark.skip(
    reason="Looks like pseudo_irradiance_datapipe is using local changes not yet merged"
)
def test_pseudo_irradiance_datapipe():
    filename = os.path.join(os.path.dirname(ocf_datapipes.__file__), "../tests/config/test.yaml")
    datapipe = pseudo_irradiance_datapipe(filename, use_nwp=True, size=32)

    batch = next(iter(datapipe))
    batch = (torch.Tensor(batch[0]), torch.Tensor(batch[1]), torch.Tensor(batch[2]))
    x = np.nan_to_num(batch[0])
    assert np.isfinite(x).all()
    assert not np.isnan(batch[1]).any()
    assert np.isfinite(batch[2]).all()
