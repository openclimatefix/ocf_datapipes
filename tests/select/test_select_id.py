import ocf_datapipes  # noqa
from ocf_datapipes.config.model import Configuration
from ocf_datapipes.load import OpenConfiguration, OpenPVFromNetCDF


def test_select_id(configuration_with_pv_netcdf):
    # load configuration
    config_datapipe = OpenConfiguration(configuration_with_pv_netcdf)
    configuration: Configuration = next(iter(config_datapipe))

    pv_datapipe = OpenPVFromNetCDF(pv=configuration.input_data.pv)

    pv_datapipe, pv_location_datapipe = pv_datapipe.fork(2)
    location_datapipe = pv_location_datapipe.location_picker()

    pv_datapipe = pv_datapipe.select_id(location_datapipe=location_datapipe, data_source_name="pv")

    data = next(iter(pv_datapipe))
