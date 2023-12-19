from ocf_datapipes.transform.numpy_batch import AddFourierSpaceTime, SaveT0Time


def test_save_t0_time(combined_datapipe):
    combined_datapipe = AddFourierSpaceTime(combined_datapipe, n_fourier_features_per_dim=2)
    combined_datapipe = SaveT0Time(combined_datapipe)
    data = next(iter(combined_datapipe))
