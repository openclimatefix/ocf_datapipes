import numpy as np

from ocf_datapipes.transform.xarray import PVFillNightNans


def test_pv_power_remove_data(passiv_datapipe):

    data_before = next(iter(passiv_datapipe))

    passiv_datapipe = PVFillNightNans(passiv_datapipe)
    data_after = next(iter(passiv_datapipe))

    assert data_before[:, 0].sum() > 0
    assert data_after[:, 0].sum() > 0
    assert data_before[:, 1].sum() > 0
    assert data_after[:, 1].sum() > 0

    assert np.isnan(data_before.values[-1, -1])
    assert not np.isnan(data_after.values[-1, -1])
