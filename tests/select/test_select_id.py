import ocf_datapipes  # noqa
from ocf_datapipes.config.model import Configuration
from ocf_datapipes.load import OpenConfiguration, OpenNWPID, OpenPVFromNetCDF
import pytest


@pytest.mark.skip("Too long")
def test_select_id(configuration_with_pv_parquet, nwp_data_with_id_filename):

    # load configuration
    config_datapipe = OpenConfiguration(configuration_with_pv_parquet)
    configuration: Configuration = next(iter(config_datapipe))

    pv_location_datapipe = OpenPVFromNetCDF(pv=configuration.input_data.pv)

    nwp_datapipe = OpenNWPID(netcdf_path=nwp_data_with_id_filename)

    location_datapipe = pv_location_datapipe.location_picker()

    nwp_datapipe = nwp_datapipe.select_id(
        location_datapipe=location_datapipe,
    )
    data = next(iter(nwp_datapipe))
    assert data.id is not None
