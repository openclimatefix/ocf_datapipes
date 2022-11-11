import numpy as np

from ocf_datapipes.transform.numpy import ChangeFloat32
from ocf_datapipes.utils.consts import BatchKey


def test_change_to_float32(all_loc_np_datapipe):
    pass

    all_loc_np_datapipe = ChangeFloat32(all_loc_np_datapipe)
    data = next(iter(all_loc_np_datapipe))

    assert data[BatchKey.gsp].dtype == np.float32
