from typing import Iterable

import numpy as np
from torchdata.datapipes import functional_datapipe
from torchdata.datapipes.iter import IterDataPipe


@functional_datapipe("process_metnet")
class ProcessMetNetIterDataPipe(IterDataPipe):
    """
    Performs the MetNet preprocessing of mean pooling Sat channels, followed by
    concatenating the center crop and mean pool

    In the paper, the radar data is space2depth'd, while satellite channel is mean pooled, but for this different
    task, we choose to do either option for satellites
    Args:
        sat_channels: Number of satellite channels
        crop_size: Center crop size
        use_space2depth: Whether to use space2depth on satellite channels, or mean pooling, like in paper

    """

    def __init__(
        self,
        source_datapipe: IterDataPipe,
        center_crop_height_pixels: int,
        center_crop_width_pixels: int,
    ):
        self.source_datapipe = source_datapipe
        self.center_crop_height_pixels = center_crop_height_pixels
        self.center_crop_width_pixels = center_crop_width_pixels

    def __iter__(self):
        for np_batch in self.source_datapipes:
            # TODO Do the processing like described
            yield np_batch
