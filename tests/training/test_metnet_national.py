import os

import torch

import ocf_datapipes
from ocf_datapipes.training.metnet_national import metnet_national_datapipe


def test_metnet_datapipe():
    filename = os.path.join(os.path.dirname(ocf_datapipes.__file__), "../tests/config/test.yaml")
    gsp_datapipe = metnet_national_datapipe(filename, use_nwp=False)

    batch = next(iter(gsp_datapipe))
    assert all(torch.isfinite(batch[0]))
    assert all(torch.isfinite(batch[1]))
