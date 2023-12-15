from ocf_datapipes.transform.numpy_batch import EncodeSpaceTime


def test_encode_space_time(combined_datapipe):
    combined_datapipe = EncodeSpaceTime(combined_datapipe, n_fourier_features_per_dim=2)
    data = next(iter(combined_datapipe))
