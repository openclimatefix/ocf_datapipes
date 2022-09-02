from datetime import timedelta

from ocf_datapipes.convert import ConvertGSPToNumpyBatch
from ocf_datapipes.transform.xarray import AddT0IdxAndSamplePeriodDuration
from ocf_datapipes.utils.consts import BatchKey


def test_convert_gsp_to_numpy_batch(gsp_datapipe):
    gsp_datapipe = AddT0IdxAndSamplePeriodDuration(
        gsp_datapipe, sample_period_duration=timedelta(minutes=5), history_duration=timedelta(minutes=60)
    )
    gsp_datapipe = ConvertGSPToNumpyBatch(gsp_datapipe)
    data = next(iter(gsp_datapipe))
    assert BatchKey.gsp in data
    assert BatchKey.gsp_id in data
