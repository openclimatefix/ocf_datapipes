"""Convert GSP to Numpy Array"""
import logging

import numpy as np
from torchdata.datapipes import functional_datapipe
from torchdata.datapipes.iter import IterDataPipe

logger = logging.getLogger(__name__)


@functional_datapipe("convert_gsp_to_numpy")
class ConvertGSPToNumpyIterDataPipe(IterDataPipe):
    """Convert GSP Xarray to Numpy Array"""

    def __init__(self, source_datapipe: IterDataPipe, return_id: bool = False):
        """
        Convert GSP Xarray to Numpy Array object

        Args:
            source_datapipe: Datapipe emitting GSP Xarray object
            return_id: Return GSP ID as well
        """
        super().__init__()
        self.source_datapipe = source_datapipe
        self.return_id = return_id

    def __iter__(self) -> np.ndarray:
        """Convert from Xarray to Numpy array"""
        logger.debug("Converting GSP to numpy")
        for xr_data in self.source_datapipe:
            returned_values = [xr_data.values]
            if self.return_id:
                pv_system_ids = xr_data["gsp_id"].values
                returned_values.append(pv_system_ids)
                yield returned_values
            else:
                yield returned_values[0]
