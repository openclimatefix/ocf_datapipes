"""Configuration Loader"""

import logging

from torch.utils.data.datapipes._decorator import functional_datapipe
from torch.utils.data.datapipes.datapipe import IterDataPipe

from ocf_datapipes.config.load import load_yaml_configuration

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

        configuration = load_yaml_configuration(self.configuration_filename)

        while True:
            yield configuration
