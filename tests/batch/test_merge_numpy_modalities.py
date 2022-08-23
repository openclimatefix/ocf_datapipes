from ocf_datapipes.batch import MergeNumpyExamplesToBatch, MergeNumpyModalities
from ocf_datapipes.convert import ConvertPVToNumpyBatch, ConvertSatelliteToNumpyBatch, ConvertNWPToNumpyBatch, ConvertGSPToNumpyBatch
from ocf_datapipes.transform.xarray import AddT0IdxAndSamplePeriodDuration
from datetime import timedelta

def test_merge_modalities(sat_hrv_dp, nwp_dp, gsp_dp, passiv_dp):
    batch_size = 4

    sat_hrv_dp = AddT0IdxAndSamplePeriodDuration(sat_hrv_dp, sample_period_duration=timedelta(minutes=5), history_duration=timedelta(hours=1))
    sat_hrv_dp = ConvertSatelliteToNumpyBatch(sat_hrv_dp, is_hrv=True)
    sat_hrv_dp = MergeNumpyExamplesToBatch(sat_hrv_dp, n_examples_per_batch=batch_size)

    nwp_dp = AddT0IdxAndSamplePeriodDuration(nwp_dp, sample_period_duration=timedelta(minutes=30), history_duration=timedelta(hours=1))
    nwp_dp = ConvertNWPToNumpyBatch(nwp_dp)
    nwp_dp = MergeNumpyExamplesToBatch(nwp_dp, n_examples_per_batch=batch_size)

    gsp_dp = AddT0IdxAndSamplePeriodDuration(gsp_dp, sample_period_duration=timedelta(hours=1), history_duration=timedelta(hours=2))
    gsp_dp = ConvertGSPToNumpyBatch(gsp_dp)
    gsp_dp = MergeNumpyExamplesToBatch(gsp_dp, n_examples_per_batch=batch_size)

    passiv_dp = AddT0IdxAndSamplePeriodDuration(passiv_dp, sample_period_duration=timedelta(minutes=5), history_duration=timedelta(hours=1))
    passiv_dp = ConvertPVToNumpyBatch(passiv_dp)
    passiv_dp = MergeNumpyExamplesToBatch(passiv_dp, n_examples_per_batch=batch_size)

    combined_dp = MergeNumpyModalities([sat_hrv_dp, passiv_dp])
    data = next(iter(combined_dp))
    print(data)


