"""Ensure there are N NWP variables by tiling the data"""
import numpy as np
from torchdata.datapipes import functional_datapipe
from torchdata.datapipes.iter import IterDataPipe

from ocf_datapipes.utils.consts import BatchKey, NumpyBatch


@functional_datapipe("ensure_n_nwp_variables")
class EnsureNNWPVariables(IterDataPipe):
    """Ensure there are N NWP variables by tiling the data"""

    def __init__(self, source_datapipe: IterDataPipe, num_variables: int):
        """
        Ensure there are N NWP variables by tiling the data

        Args:
            source_datapipe: NumpyBatch emitting NWP datapipe
            num_variables: Number of variables needed
        """
        self.source_datapipe = source_datapipe
        self.num_variables = num_variables

    def __iter__(self) -> NumpyBatch:
        """Ensure there are N NWP variables by tiling the data"""
        for np_batch in self.source_datapipe:
            num_tiles = int(np.ceil(self.num_variables / len(np_batch[BatchKey.nwp_channel_names])))
            np_batch[BatchKey.nwp_channel_names] = np.tile(
                np_batch[BatchKey.nwp_channel_names], num_tiles
            )[: self.num_variables]
            np_batch[BatchKey.nwp] = np.tile(np_batch[BatchKey.nwp], (1, 1, num_tiles, 1, 1))[
                :, :, : self.num_variables, :, :
            ]
            yield np_batch
