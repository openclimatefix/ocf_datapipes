import fsspec
from pathy import Pathy
from pyaml_env import parse_config
from torchdata.datapipes import functional_datapipe
from torchdata.datapipes.iter import IterDataPipe


@functional_datapipe("open_config")
class OpenConfigurationIterDataPipe(IterDataPipe):
    def __init__(self, configuration_filename: str):
        self.configuration_filename = configuration_filename

    def __iter__(self):
        with fsspec.open(self.configuration_filename, mode="r") as stream:
            configuration = parse_config(data=stream)

        # TODO Load into Pydantic Configuration class

        while True:
            yield configuration
