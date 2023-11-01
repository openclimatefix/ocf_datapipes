"""Load ASOS data from local files for training/inference"""
import io
import logging
from datetime import datetime
from pathlib import Path
from typing import Optional, Union

import fsspec
import numpy as np
import pandas as pd
import xarray as xr
from torchdata.datapipes import functional_datapipe
from torchdata.datapipes.iter import IterDataPipe
from ocf_datapipes.config.model import Sensor

_log = logging.getLogger(__name__)

@functional_datapipe("OpenAWOS")
class OpenAWOSFromNetCDFIterDataPipe(IterDataPipe):
    """OpenAWOSFromNetCDFIterDataPipe"""

    def __init__(
        self,
        sensor: Sensor,
    ):
        """
        Datapipe to load sensor data from AWOS network

        Args:
            sensor: Sensor configuration
        """
        super().__init__()
        self.sensor = sensor
        self.filename = self.sensor.sensor_filename

    def __iter__(self):
        with fsspec.open(self.filename, "rb") as f:
            ds = xr.open_dataset(f)
        while True:
            yield ds
