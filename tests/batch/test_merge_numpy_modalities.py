from datetime import timedelta

from ocf_datapipes.batch import MergeNumpyModalities
from ocf_datapipes.convert import (
    ConvertGSPToNumpyBatch,
    ConvertNWPToNumpyBatch,
    ConvertPVToNumpyBatch,
    ConvertSatelliteToNumpyBatch,
)
from ocf_datapipes.transform.xarray import AddT0IdxAndSamplePeriodDuration


def test_merge_modalities(sat_hrv_datapipe, nwp_datapipe, gsp_datapipe, passiv_datapipe):
    batch_size = 4

    sat_hrv_datapipe = AddT0IdxAndSamplePeriodDuration(
        sat_hrv_datapipe,
        sample_period_duration=timedelta(minutes=5),
        history_duration=timedelta(hours=1),
    )
    sat_hrv_datapipe = ConvertSatelliteToNumpyBatch(sat_hrv_datapipe, is_hrv=True)

    nwp_datapipe = AddT0IdxAndSamplePeriodDuration(
        nwp_datapipe,
        sample_period_duration=timedelta(minutes=30),
        history_duration=timedelta(hours=1),
    )
    nwp_datapipe = ConvertNWPToNumpyBatch(nwp_datapipe)

    gsp_datapipe = AddT0IdxAndSamplePeriodDuration(
        gsp_datapipe, sample_period_duration=timedelta(hours=1), history_duration=timedelta(hours=2)
    )
    gsp_datapipe = ConvertGSPToNumpyBatch(gsp_datapipe)

    passiv_datapipe = AddT0IdxAndSamplePeriodDuration(
        passiv_datapipe,
        sample_period_duration=timedelta(minutes=5),
        history_duration=timedelta(hours=1),
    )
    passiv_datapipe = ConvertPVToNumpyBatch(passiv_datapipe)

    combined_datapipe = MergeNumpyModalities([sat_hrv_datapipe, passiv_datapipe]).batch(batch_size)
    data = next(iter(combined_datapipe))
    print(data)
