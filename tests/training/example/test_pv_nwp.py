from ocf_datapipes.training.example.pv_nwp import pv_nwp_datapipe
from ocf_datapipes.batch import BatchKey


def test_pv_nwp_datapipe(configuration_with_pv_netcdf_and_nwp):
    datapipe = pv_nwp_datapipe(configuration_with_pv_netcdf_and_nwp)

    batch = next(iter(datapipe))
    assert len(batch[BatchKey.pv_time_utc][0]) == 19
