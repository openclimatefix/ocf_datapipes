from ocf_datapipes.convert import ConvertGSPToNumpyBatch
from ocf_datapipes.transform.xarray import AddT0IdxAndSamplePeriodDuration

from datetime import timedelta



def test_convert_gsp_to_numpy_batch(gsp_dp):
    gsp_dp = AddT0IdxAndSamplePeriodDuration(
        gsp_dp, sample_period_duration=timedelta(minutes=5), history_duration=timedelta(minutes=60)
    )
    gsp_dp = ConvertGSPToNumpyBatch(gsp_dp)
    data = next(iter(gsp_dp))
    assert data is not None
