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

_log = logging.getLogger(__name__)

@functional_datapipe("OpenAWOS")
class OpenAWOSFromNetCDFIterDataPipe(IterDataPipe):
    """OpenAWOSFromNetCDFIterDataPipe

    Args:
        filename: Path to the NetCDF file
        tag: Tag for train or test
    """

    def __init__(
        self,
        filename: Union[Path, str],
    ):
        super().__init__()
        self.filename = filename

    def __iter__(self):
        with fsspec.open(self.filename, "rb") as f:
            ds = xr.open_dataset(f)
        return ds
