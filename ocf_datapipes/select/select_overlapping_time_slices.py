from torchdata.datapipes.iter import IterDataPipe, Zipper
from torchdata.datapipes import functional_datapipe
from typing import Union, Iterable


@functional_datapipe("select_overlapping_time_slice")
class SelectOverlappingTimeSlice(IterDataPipe):
    def __init__(self, source_dps: Iterable[IterDataPipe]):
        super().__init__()
        self.source_dps = source_dps

    def __iter__(self):
        for set_of_xrs in Zipper(*self.source_dps):
            pass
