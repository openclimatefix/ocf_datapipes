from ocf_datapipes.training.gsp_national import gsp_national_datapipe
from ocf_datapipes.utils.consts import BatchKey


def test_nwp_pv_datapipe(configuration_with_gsp_and_nwp):

    gsp_datapipe = gsp_national_datapipe(configuration_with_gsp_and_nwp)

    batch = next(iter(gsp_datapipe))
    print('xxx')
    print(batch)
    print('xxx')
    assert len(batch[BatchKey.gsp_time_utc][0]) == 19
