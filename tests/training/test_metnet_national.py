import os

import ocf_datapipes
from ocf_datapipes.training.metnet_national import metnet_national_datapipe
from ocf_datapipes.utils.consts import BatchKey


def test_metnet_datapipe():
    filename = os.path.join(os.path.dirname(ocf_datapipes.__file__), "../tests/config/test.yaml")
    gsp_datapipe = metnet_national_datapipe(filename)

    batch = next(iter(gsp_datapipe))
    print(batch[0].shape)
    print(batch[1].shape)
