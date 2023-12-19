from ocf_datapipes.training.example.gsp_national import gsp_national_datapipe
from ocf_datapipes.batch import BatchKey


def test_gsp_national_datapipe(configuration_with_gsp_and_nwp):
    datapipe = gsp_national_datapipe(configuration_with_gsp_and_nwp)

    batch = next(iter(datapipe))

    # 4 in past, now, and 2 in the future
    assert len(batch[BatchKey.gsp_time_utc][0]) == 7
