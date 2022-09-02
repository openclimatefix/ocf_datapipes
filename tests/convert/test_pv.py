from datetime import timedelta

from ocf_datapipes.convert import ConvertPVToNumpyBatch
from ocf_datapipes.transform.xarray import AddT0IdxAndSamplePeriodDuration
from ocf_datapipes.utils.consts import BatchKey


def test_convert_passiv_to_numpy_batch(passiv_datapipe):
    passiv_datapipe = AddT0IdxAndSamplePeriodDuration(
        passiv_datapipe,
        sample_period_duration=timedelta(minutes=5),
        history_duration=timedelta(minutes=60),
    )
    passiv_datapipe = ConvertPVToNumpyBatch(passiv_datapipe)
    data = next(iter(passiv_datapipe))
    assert BatchKey.pv in data
    assert BatchKey.pv_t0_idx in data


def test_convert_pvoutput_to_numpy_batch(pvoutput_datapipe):
    pvoutput_datapipe = AddT0IdxAndSamplePeriodDuration(
        pvoutput_datapipe,
        sample_period_duration=timedelta(minutes=5),
        history_duration=timedelta(minutes=60),
    )

    pvoutput_datapipe = ConvertPVToNumpyBatch(pvoutput_datapipe)

    data = next(iter(pvoutput_datapipe))
    assert BatchKey.pv in data
    assert BatchKey.pv_t0_idx in data
