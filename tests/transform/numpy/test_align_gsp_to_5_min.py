from ocf_datapipes.transform.numpy import AlignGSPto5Min
from ocf_datapipes.batch import MergeNumpyModalities
from ocf_datapipes.utils.consts import BatchKey

def test_align_gsp_to_5_min(sat_hrv_np_dp, gsp_np_dp):
    combined_dp = MergeNumpyModalities([sat_hrv_np_dp, gsp_np_dp])
    combined_dp = AlignGSPto5Min(combined_dp, batch_key_for_5_min_datetimes=BatchKey.hrvsatellite_time_utc)
    data = next(iter(combined_dp))

