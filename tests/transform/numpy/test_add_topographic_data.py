from datetime import timedelta

from ocf_datapipes.batch import MergeNumpyExamplesToBatch
from ocf_datapipes.convert import ConvertSatelliteToNumpyBatch
from ocf_datapipes.transform.numpy import AddTopographicData
from ocf_datapipes.transform.xarray import AddT0IdxAndSamplePeriodDuration, ReprojectTopography


def test_add_topo_data_hrvsatellite(sat_hrv_dp, topo_dp):
    sat_hrv_dp = AddT0IdxAndSamplePeriodDuration(
        sat_hrv_dp, sample_period_duration=timedelta(minutes=5), history_duration=timedelta(hours=1)
    )
    sat_hrv_dp = ConvertSatelliteToNumpyBatch(sat_hrv_dp, is_hrv=True)
    sat_hrv_dp = MergeNumpyExamplesToBatch(sat_hrv_dp, n_examples_per_batch=4)
    topo_dp = ReprojectTopography(topo_dp)
    combined_dp = AddTopographicData(sat_hrv_dp, topo_dp=topo_dp)
    data = next(iter(combined_dp))
    assert data is not None
