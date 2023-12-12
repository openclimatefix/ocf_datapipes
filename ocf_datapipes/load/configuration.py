"""Configuration Loader"""
import logging

import fsspec
from pyaml_env import parse_config
from torch.utils.data.datapipes._decorator import functional_datapipe
from torch.utils.data.datapipes.datapipe import IterDataPipe

from ocf_datapipes.config.model import Configuration

logger = logging.getLogger(__name__)


def load_configuration(filename):
    """Load and return configuration file"""
    with fsspec.open(filename, mode="r") as stream:
        configuration = parse_config(data=stream)

    configuration = Configuration(**configuration)
    return configuration


@functional_datapipe("open_config")
class OpenConfigurationIterDataPipe(IterDataPipe):
    """Open and return the configuration data"""

    def __init__(self, configuration_filename: str):
        """
        Open and return config data

        Args:
            configuration_filename: Filename to open
        """
        self.configuration_filename = configuration_filename

    def __iter__(self):
        """Open and return configuration file"""
        logger.debug(f"Going to open {self.configuration_filename}")

        configuration = load_configuration(self.configuration_filename)

        while True:
            yield configuration
