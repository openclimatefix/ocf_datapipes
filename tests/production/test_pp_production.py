import os

from freezegun import freeze_time

import ocf_datapipes
from ocf_datapipes.production.power_perceiver import power_perceiver_production_datapipe
from ocf_datapipes.utils.consts import BatchKey


@freeze_time("2022-01-01 08:00")
def test_pp_production_datapipe(pv_yields_and_systems):

    filename = os.path.join(os.path.dirname(ocf_datapipes.__file__), "../tests/config/test.yaml")

    pp_dp = power_perceiver_production_datapipe(filename)

    batch = next(iter(pp_dp))

    assert len(batch[BatchKey.hrvsatellite_time_utc]) == 4
    assert len(batch[BatchKey.hrvsatellite_time_utc][0]) == 19  # 6 history + now + 12 future = 19
    assert len(batch[BatchKey.nwp_target_time_utc][0]) == 3
    assert len(batch[BatchKey.nwp_init_time_utc][0]) == 3
    assert len(batch[BatchKey.pv_time_utc][0]) == 19
    assert len(batch[BatchKey.gsp_time_utc][0]) == 4  # 1 history + now + 2 future = 4

    assert batch[BatchKey.hrvsatellite_actual].shape == (4, 7, 1, 64, 64)  # 2nd dim is 6 history + now
    assert batch[BatchKey.nwp].shape == (4, 3, 1, 2, 2)  # 2nd dim is 1 history + now + 1 future
    assert batch[BatchKey.pv].shape == (4, 7, 32)  # 2nd dim is 6 history + now
    assert batch[BatchKey.gsp].shape == (4, 2, 1)  # 2nd dim is 1 history + now
    assert batch[BatchKey.hrvsatellite_surface_height].shape == (4, 64, 64)
