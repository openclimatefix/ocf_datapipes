from ocf_datapipes.transform.numpy import EncodeSpaceTime, SaveT0Time


def test_save_t0_time(combined_datapipe):
    combined_datapipe = EncodeSpaceTime(combined_datapipe, n_fourier_features_per_dim=2)
    combined_datapipe = SaveT0Time(combined_datapipe)
    data = next(iter(combined_datapipe))