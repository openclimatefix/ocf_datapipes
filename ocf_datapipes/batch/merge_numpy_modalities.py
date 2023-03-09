"""Merge multiple modalities together in NumpyBatch"""
from torchdata.datapipes import functional_datapipe
from torchdata.datapipes.iter import IterDataPipe

from ocf_datapipes.utils import Zipper
from ocf_datapipes.utils.consts import NumpyBatch
from ocf_datapipes.utils.utils import profile


@functional_datapipe("merge_numpy_modalities")
class MergeNumpyModalitiesIterDataPipe(IterDataPipe):
    """Merge multiple modalities together in NumpyBatch"""

    def __init__(self, source_datapipes: [IterDataPipe]):
        """
        Merge multiple modalities together in NumpyBatch

        Args:
            source_datapipes: Set of datapipes to merge emitting NumpyBatch objects
        """
        self.zipped_source_datapipes = Zipper(*source_datapipes)
    
    def __len__(self):
        return len(self.zipped_source_datapipes)
    
    def __iter__(self) -> NumpyBatch:
        """Merge multiple modalities together in NumpyBatch"""
        for np_batches in self.zipped_source_datapipes:
            with profile("merging all datapipes"):
                example: NumpyBatch = {}
                for np_batch in np_batches:
                    example.update(np_batch)
                yield example
