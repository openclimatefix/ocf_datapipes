"""Offset T0 for training to more realistically match production"""
from typing import Union

import numpy as np
import pandas as pd
from torchdata.datapipes import functional_datapipe
from torchdata.datapipes.iter import IterDataPipe


@functional_datapipe("offset_t0")
class OffsetT0IterDataPipe(IterDataPipe):
    """Offset T0 for training to more realistically match production"""

    def __init__(
        self,
        source_datapipe: IterDataPipe,
        max_t0_offset_minutes: Union[float, int],
        min_t0_offset_minutes: Union[float, int] = 0.0,
    ):
        """
        Offset T0 for more realistic training

        Args:
            source_datapipe: Datapipe of t0 times
            max_t0_offset_minutes: Max offset to be applied
            min_t0_offset_minutes: Min offset to be applied
        """
        self.source_datapipe = source_datapipe
        self.max_t0_offset_minutes = max_t0_offset_minutes
        self.min_t0_offset_minutes = min_t0_offset_minutes

    def __iter__(self):
        for t0 in self.source_datapipe:
            offset = pd.Timedelta(
                minutes=np.random.randint(self.min_t0_offset_minutes, self.max_t0_offset_minutes)
            )
            yield t0 - offset
