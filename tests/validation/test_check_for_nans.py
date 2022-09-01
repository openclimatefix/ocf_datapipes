import numpy as np
import pytest

from ocf_datapipes.validation import CheckNaNs


def test_check_nans_fail_dataarray(topo_dp):
    topo_dp = CheckNaNs(topo_dp)
    with pytest.raises(Exception):
        next(iter(topo_dp))


def test_check_nans_fill(topo_dp):
    topo_dp = CheckNaNs(topo_dp, fill_nans=True)
    data = next(iter(topo_dp))
    assert np.all(data != np.nan)
