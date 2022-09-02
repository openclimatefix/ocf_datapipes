from ocf_datapipes.batch import MergeNumpyModalities
from ocf_datapipes.transform.numpy import AlignGSPto5Min, EncodeSpaceTime
from ocf_datapipes.utils.consts import BatchKey


def test_encode_space_time(sat_hrv_np_datapipe, passiv_np_datapipe, gsp_np_datapipe):
    combined_datapipe = MergeNumpyModalities(
        [sat_hrv_np_datapipe, gsp_np_datapipe, passiv_np_datapipe]
    )
    combined_datapipe = AlignGSPto5Min(
        combined_datapipe, batch_key_for_5_min_datetimes=BatchKey.hrvsatellite_time_utc
    )
    combined_datapipe = EncodeSpaceTime(combined_datapipe, n_fourier_features_per_dim=2)
    data = next(iter(combined_datapipe))
