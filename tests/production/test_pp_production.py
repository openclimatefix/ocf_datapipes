from ocf_datapipes.production.power_perceiver import power_perceiver_production_datapipe
from ocf_datapipes.utils.consts import BatchKey

import ocf_datapipes
import os
import pytest


@pytest.mark.skip("Need to set up laod PV from database first")
def test_pp_production_datapipe():

    filename = os.path.join(os.path.dirname(ocf_datapipes.__file__), "../tests/config/test.yaml")

    pp_dp = power_perceiver_production_datapipe(filename)

    batch = next(iter(pp_dp))

    assert len(batch[BatchKey.hrvsatellite_time_utc]) == 4
    assert len(batch[BatchKey.hrvsatellite_time_utc][0]) == 37
    assert len(batch[BatchKey.nwp_target_time_utc][0]) == 6
    assert len(batch[BatchKey.nwp_init_time_utc][0]) == 6
    assert len(batch[BatchKey.pv_time_utc][0]) == 37
    assert len(batch[BatchKey.gsp_time_utc][0]) == 21

    assert batch[BatchKey.hrvsatellite_actual].shape == (4, 13, 1, 128, 256)
    assert batch[BatchKey.nwp].shape == (4, 6, 1, 4, 4)
    assert batch[BatchKey.pv].shape == (4, 13, 8)
    assert batch[BatchKey.gsp].shape == (4, 5, 1)
    assert batch[BatchKey.hrvsatellite_surface_height].shape == (4, 128, 256)
