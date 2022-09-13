"""Add fake future data for proper Power Perceiver Production"""
import numpy as np
from torchdata.datapipes import functional_datapipe
from torchdata.datapipes.iter import IterDataPipe

from ocf_datapipes.utils.consts import BatchKey, NumpyBatch


@functional_datapipe("add_zeroed_future_data")
class AddZeroedFutureDataIterDataPipe(IterDataPipe):
    """Extends timestamps into the future"""

    def __init__(self, source_datapipe: IterDataPipe, key: BatchKey, time_key: BatchKey):
        """
        Extends data into the future, zeroed out

        This assumes that the current time_utc array only covers history + now,
        so just extends it further into the future

        Args:
            source_datapipe: Datapipe of NumpyBatch
            key: BatchKey to extend
            time_key: Batch key to use as the time for how many to append
        """
        self.source_datapipe = source_datapipe
        self.key = key
        self.time_key = time_key

    def __iter__(self) -> NumpyBatch:
        for np_batch in self.source_datapipe:
            data = np_batch[self.key]
            new_data = np.zeros((len(np_batch[self.time_key]), *data.shape[1:]), dtype=data.dtype)
            for t in range(data.shape[0]):
                new_data[t] = data[t]
            np_batch[self.key] = new_data
            yield np_batch
