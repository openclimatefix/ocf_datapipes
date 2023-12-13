from ocf_datapipes.select import SelectGSPIDs


def test_select_non_national(gsp_datapipe):
    drop_pipe = SelectGSPIDs(gsp_datapipe)
    assert 0 not in next(iter(drop_pipe)).gsp_id
