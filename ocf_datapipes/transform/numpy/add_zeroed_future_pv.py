from torchdata.datapipes.iter import IterDataPipe
from torchdata.datapipes import functional_datapipe

from ocf_datapipes.utils.consts import NumpyBatch, BatchKey
import numpy as np

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
        """
        self.source_datapipe = source_datapipe
        self.key = key
        self.time_key = time_key

    def __iter__(self) -> NumpyBatch:
        for np_batch in self.source_datapipe:
            data = np_batch[self.key]
            new_data = np.zeros((len(np_batch[self.time_key]), *data.shape[1:]), dtype=data.dtype)
            for t in range(data.shape[1]):
                new_data[t] = data[t]
            np_batch[self.key] = new_data
            yield np_batch
