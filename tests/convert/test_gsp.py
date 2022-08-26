from ocf_datapipes.convert import ConvertGSPToNumpyBatch


def test_convert_gsp_to_numpy_batch(gsp_dp):
    gsp_dp = ConvertGSPToNumpyBatch(gsp_dp, is_hrv=False)
    data = next(iter(gsp_dp))
    assert data is not None
