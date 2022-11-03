from ocf_datapipes.select import DropGSP


def test_drop_non_national(gsp_datapipe):
    drop_pipe = DropGSP(gsp_datapipe)
    assert 0 not in next(iter(drop_pipe)).gsp_id
