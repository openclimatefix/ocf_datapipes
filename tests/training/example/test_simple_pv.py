from ocf_datapipes.training.example.simple_pv import simple_pv_datapipe
from ocf_datapipes.utils.consts import BatchKey


def test_simple_pv_datapipe(configuration_with_pv_netcdf):
    datapipe = simple_pv_datapipe(configuration_with_pv_netcdf).set_length(2)

    batch = next(iter(datapipe))

    assert len(batch[BatchKey.pv_time_utc][0]) == 19
