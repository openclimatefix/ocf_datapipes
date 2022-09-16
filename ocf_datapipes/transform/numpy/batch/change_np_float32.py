"""Datapipes to add Sun position to NumpyBatch"""

import logging

import numpy as np
from torchdata.datapipes import functional_datapipe
from torchdata.datapipes.iter import IterDataPipe

logger = logging.getLogger(__name__)


@functional_datapipe("change_float32")
class ChangeFloat32IterDataPipe(IterDataPipe):
    """Change to float 64 to 32s"""

    def __init__(self, source_datapipe: IterDataPipe):
        """
        Change float 64s to float 32s

        Args:
            source_datapipe: Datapipe of NumpyBatch
        """
        self.source_datapipe = source_datapipe

    def __iter__(self):
        for np_batch in self.source_datapipe:

            logger.debug("Changing arrays to float32s")

            for key in np_batch.keys():
                if isinstance(np_batch[key], np.ndarray):
                    if np_batch[key].dtype == np.float64:
                        np_batch[key] = np_batch[key].astype("float32")

            yield np_batch
