from ocf_datapipes.batch import MergeNumpyModalities
from ocf_datapipes.transform.numpy import AlignGSPto5Min
from ocf_datapipes.utils.consts import BatchKey


def test_align_gsp_to_5_min(sat_hrv_np_datapipe, gsp_np_datapipe):
    combined_datapipe = MergeNumpyModalities([sat_hrv_np_datapipe, gsp_np_datapipe])
    combined_datapipe = AlignGSPto5Min(
        combined_datapipe, batch_key_for_5_min_datetimes=BatchKey.hrvsatellite_time_utc
    )
    data = next(iter(combined_datapipe))
