from typing import Union

import numpy as np
from torchdata.datapipes import functional_datapipe
from torchdata.datapipes.iter import IterDataPipe


@functional_datapipe("offset_t0")
class OffsetT0IterDataPipe(IterDataPipe):
    def __init__(
        self,
        source_dp: IterDataPipe,
        max_t0_offset_minutes: Union[float, int],
        min_t0_offset_minutes: Union[float, int] = 0.0,
    ):
        self.source_dp = source_dp
        self.max_t0_offset_minutes = max_t0_offset_minutes
        self.min_t0_offset_minutes = min_t0_offset_minutes

    def __iter__(self):
        pass
