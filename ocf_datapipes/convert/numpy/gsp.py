"""Convert GSP to Numpy Array"""
import logging

import numpy as np
from torchdata.datapipes import functional_datapipe
from torchdata.datapipes.iter import IterDataPipe

logger = logging.getLogger(__name__)


@functional_datapipe("convert_gsp_to_numpy")
class ConvertGSPToNumpyIterDataPipe(IterDataPipe):
    """Convert GSP Xarray to Numpy Array"""

    def __init__(self, source_datapipe: IterDataPipe):
        """
        Convert GSP Xarray to Numpy Array object

        Args:
            source_datapipe: Datapipe emitting GSP Xarray object
        """
        super().__init__()
        self.source_datapipe = source_datapipe

    def __iter__(self) -> np.ndarray:
        """Convert from Xarray to Numpy array"""
        logger.debug("Converting GSP to numpy")
        for xr_data in self.source_datapipe:
            yield xr_data.values
