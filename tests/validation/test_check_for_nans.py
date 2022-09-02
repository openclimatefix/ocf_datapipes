import numpy as np
import pytest

from ocf_datapipes.validation import CheckNaNs


def test_check_nans_fail_dataarray(topo_datapipe):
    topo_datapipe = CheckNaNs(topo_datapipe)
    with pytest.raises(Exception):
        next(iter(topo_datapipe))


def test_check_nans_fill(topo_datapipe):
    topo_datapipe = CheckNaNs(topo_datapipe, fill_nans=True)
    data = next(iter(topo_datapipe))
    assert np.all(data != np.nan)
