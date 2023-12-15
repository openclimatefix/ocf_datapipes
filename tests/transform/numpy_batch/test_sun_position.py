from ocf_datapipes.transform.numpy_batch import AddSunPosition


def test_add_sun_position_pv(combined_datapipe):
    combined_datapipe = AddSunPosition(combined_datapipe, modality_name="pv")
    data = next(iter(combined_datapipe))


def test_add_sun_position_gsp(combined_datapipe):
    combined_datapipe = AddSunPosition(combined_datapipe, modality_name="gsp")
    data = next(iter(combined_datapipe))
