from ocf_datapipes.training.gsp_pv_satellite_nwp import gsp_pv_nwp_satellite_data_pipeline
from ocf_datapipes.utils.consts import BatchKey


def test_gsp_pv_nwp_satellite_datapipe(configuration_filename):

    gsp_datapipe = gsp_pv_nwp_satellite_data_pipeline(configuration_filename).set_length(2)

    batch = next(iter(gsp_datapipe))

    # 4 in past, now, and 2 in the future
    print(batch)
    assert len(batch[BatchKey.gsp_time_utc][0]) == 7
