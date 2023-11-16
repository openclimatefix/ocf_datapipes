import os

import numpy as np
import pytest
from torch.utils.data import DataLoader

import ocf_datapipes
from ocf_datapipes.training.metnet_gsp_national import metnet_national_datapipe


@pytest.mark.skip("Skip as takes too long")
def test_metnet_gsp_national_datapipe():
    filename = os.path.join(os.path.dirname(ocf_datapipes.__file__), "../tests/config/test.yaml")
    datapipe = metnet_national_datapipe(filename, use_pv=False)
    dataloader = DataLoader(datapipe)
    for i, batch in enumerate(dataloader):
        _ = batch
        if i + 1 % 50000 == 0:
            break


def test_metnet_gsp_national_image_datapipe():
    filename = os.path.join(os.path.dirname(ocf_datapipes.__file__), "../tests/config/test.yaml")
    datapipe = metnet_national_datapipe(filename, use_pv=False, gsp_in_image=True, output_size=128)
    dataloader = iter(datapipe)
    batch = next(dataloader)
    x, y = batch
    assert np.isfinite(x).all()
    assert np.isfinite(y).all()
