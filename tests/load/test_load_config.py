from ocf_datapipes.load import OpenConfiguration


def test_open_config(configuration_filename):
    config_datapipe = OpenConfiguration(configuration_filename)
    configuration = next(iter(config_datapipe))
