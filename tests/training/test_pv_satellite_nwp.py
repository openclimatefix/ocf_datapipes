from ocf_datapipes.training.pv_satellite_nwp import pv_nwp_satellite_data_pipeline
from ocf_datapipes.utils.consts import BatchKey

import pytest
@pytest.mark.skip("Too Memory Intensive")
def test_gsp_pv_nwp_satellite_datapipe(configuration_filename):

    pv_datapipe = pv_nwp_satellite_data_pipeline(configuration_filename)

    batch = next(iter(pv_datapipe))

    # 4 in past, now, and 2 in the future
    print(batch)
    assert len(batch[BatchKey.pv_time_utc][0]) == 37