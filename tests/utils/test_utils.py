import numpy as np
from ocf_datapipes.utils.utils import searchsorted


def test_searchsorted():
    ys = np.array([1, 2, 3, 4, 5], dtype=np.float32)
    assert searchsorted(ys, 2.1) == 2
    ys_r = np.array([5, 4, 3, 2, 1], dtype=np.float32)
    assert searchsorted(ys_r, 2.1, assume_ascending=False) == 3
