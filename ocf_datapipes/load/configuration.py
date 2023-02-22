"""Configuration Loader"""
from typing import Union
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

    def __init__(self, configuration: Union[str, dict]):
        """
        Open and return config data

        Args:
            configuration: Filename to open or already opened configuration dictionary
        """
        self.configuration = configuration

    def __iter__(self):
        """Open and return configuration file"""
        
        if isinstance(self.configuration, str):
            logger.debug(f"Going to open {self.configuration}")
            with fsspec.open(self.configuration, mode="r") as stream:
                configuration = parse_config(data=stream)
        else:
            configuration = self.configuration
        
        logger.debug(f"Converting to Configuration ({configuration})")
        configuration = Configuration(**configuration)

        while True:
            yield configuration
