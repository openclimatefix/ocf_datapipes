from torchdata.datapipes import functional_datapipe
from torchdata.datapipes.iter import IterDataPipe


@functional_datapipe("time_picker")
class TimePickerIterDataPipe(IterDataPipe):
    def __init__(self, source_datapipe: IterDataPipe):
        self.source_datapipe = source_datapipe

    def __iter__(self):
        for pd_data in self.source_datapipe:
            pass
