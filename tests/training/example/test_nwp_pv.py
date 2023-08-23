from ocf_datapipes.training.example.nwp_pv import nwp_pv_datapipe
from ocf_datapipes.utils.consts import BatchKey


def test_nwp_pv_datapipe(configuration_with_pv_netcdf_and_nwp):
    datapipe = nwp_pv_datapipe(configuration_with_pv_netcdf_and_nwp).set_length(2)

    batch = next(iter(datapipe))
    assert len(batch[BatchKey.pv_time_utc][0]) == 19
