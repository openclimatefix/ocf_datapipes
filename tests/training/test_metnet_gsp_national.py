import os

import pytest
from torchdata.dataloader2 import DataLoader2

import ocf_datapipes
from ocf_datapipes.training.metnet_gsp_national import metnet_national_datapipe


@pytest.mark.skip("Skip as takes too long")
def test_metnet_datapipe():
    filename = os.path.join(os.path.dirname(ocf_datapipes.__file__), "../tests/config/test.yaml")
    gsp_datapipe = metnet_national_datapipe(filename, use_pv=False)
    dataloader = DataLoader2(gsp_datapipe)
    for i, batch in enumerate(dataloader):
        _ = batch
        if i + 1 % 50000 == 0:
            break
