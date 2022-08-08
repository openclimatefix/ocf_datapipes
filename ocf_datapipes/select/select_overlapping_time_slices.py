from typing import Iterable, Union

from torchdata.datapipes import functional_datapipe
from torchdata.datapipes.iter import IterDataPipe, Zipper


@functional_datapipe("select_overlapping_time_slice")
class SelectOverlappingTimeSlice(IterDataPipe):
    def __init__(self, source_dps: Iterable[IterDataPipe]):
        super().__init__()
        self.source_dps = source_dps

    def __iter__(self):
        for set_of_xrs in Zipper(*self.source_dps):
            pass
