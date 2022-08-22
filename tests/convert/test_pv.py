from ocf_datapipes.convert import ConvertPVToNumpyBatch


def test_convert_passiv_to_numpy_batch(passiv_dp):
    passiv_dp = ConvertPVToNumpyBatch(passiv_dp, is_hrv=False)
    data = next(iter(passiv_dp))
    assert data is not None


def test_convert_pvoutput_to_numpy_batch(pvoutput_dp):
    pvoutput_dp = ConvertPVToNumpyBatch(pvoutput_dp, is_hrv=True)
    data = next(iter(pvoutput_dp))
    assert data is not None
