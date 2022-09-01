"""Configuration Loader"""
import logging

import fsspec
from pyaml_env import parse_config
from torchdata.datapipes import functional_datapipe
from torchdata.datapipes.iter import IterDataPipe

from ocf_datapipes.config.model import Configuration

logger = logging.getLogger(__name__)


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
        with fsspec.open(self.configuration_filename, mode="r") as stream:
            configuration = parse_config(data=stream)

        logger.debug(f"Converting to Configuration ({configuration})")
        configuration = Configuration(**configuration)

        while True:
            yield configuration
