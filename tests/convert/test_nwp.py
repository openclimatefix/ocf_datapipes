from ocf_datapipes.convert import ConvertNWPToNumpyBatch


def test_convert_nwp_to_numpy_batch(nwp_dp):
    nwp_dp = ConvertNWPToNumpyBatch(nwp_dp)
    data = next(iter(nwp_dp))
    assert data is not None
