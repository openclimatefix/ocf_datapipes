"""Merge multiple modalities together in NumpyBatch"""
from torchdata.datapipes import functional_datapipe
from torchdata.datapipes.iter import IterDataPipe, Zipper

from ocf_datapipes.utils.consts import NumpyBatch


@functional_datapipe("merge_numpy_modalities")
class MergeNumpyModalitiesIterDataPipe(IterDataPipe):
    """Merge multiple modalities together in NumpyBatch"""

    def __init__(self, source_datapipes: [IterDataPipe]):
        """
        Merge multiple modalities together in NumpyBatch

        Args:
            source_datapipes: Set of datapipes to merge emitting NumpyBatch objects
        """
        self.source_datapipes = source_datapipes

    def __iter__(self) -> NumpyBatch:
        """Merge multiple modalities together in NumpyBatch"""
        for np_batches in Zipper(*self.source_datapipes):
            example: NumpyBatch = {}
            for np_batch in np_batches:
                example.update(np_batch)
            yield example
