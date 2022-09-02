from ocf_datapipes.load import OpenConfiguration


def test_open_config():
    config_datapipe = OpenConfiguration("tests/config/test.yaml")
    configuration = next(iter(config_datapipe))
    print(configuration)
