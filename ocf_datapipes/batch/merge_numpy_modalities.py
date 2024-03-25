"""Merge multiple modalities together in NumpyBatch"""

from torch.utils.data import IterDataPipe, functional_datapipe

from ocf_datapipes.batch import BatchKey, NumpyBatch
from ocf_datapipes.utils import Zipper


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


@functional_datapipe("merge_nwp_numpy_modalities")
class MergeNWPNumpyModalitiesIterDataPipe(IterDataPipe):
    """Merge multiple NWPNumpyBatches into a NumpyBatch"""

    def __init__(self, datapipes_dict: dict[IterDataPipe]):
        """
        Merge multiple NWPNumpyBatches into a NumpyBatch

        Args:
            datapipes_dict: dict of datapipes to merge emitting NWPNumpyBatch objects
        """
        self.datapipes_dict = datapipes_dict

    def __iter__(self) -> NumpyBatch:
        """Merge multiple NWPNumpyBatches into a NumpyBatch"""
        keys = list(self.datapipes_dict.keys())
        datapipes = [self.datapipes_dict[k] for k in keys]
        for nwp_np_batches in Zipper(*datapipes):
            example: NumpyBatch = {BatchKey.nwp: {k: v for k, v in zip(keys, nwp_np_batches)}}
            yield example
