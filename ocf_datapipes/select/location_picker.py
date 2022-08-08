from torchdata.datapipes.iter import IterDataPipe
from torchdata.datapipes import functional_datapipe


@functional_datapipe("location_picker")
class LocationPickerIterDataPipe(IterDataPipe):
    def __init__(self, source_dp: IterDataPipe):
        super().__init__()
        self.source_dp = source_dp

    def __iter__(self):
        pass
