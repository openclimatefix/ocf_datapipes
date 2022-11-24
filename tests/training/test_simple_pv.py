from ocf_datapipes.training.simple_pv import simple_pv_datapipe
from ocf_datapipes.utils.consts import BatchKey

import pytest


@pytest.mark.skip("Too Memory Intensive")
def test_pp_production_datapipe(configuration_with_pv_parquet):

    pp_datapipe = simple_pv_datapipe(configuration_with_pv_parquet)

    batch = next(iter(pp_datapipe))

    assert len(batch[BatchKey.pv_time_utc][0]) == 19
