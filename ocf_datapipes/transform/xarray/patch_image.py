from torchdata.datapipes import functional_datapipe
from torchdata.datapipes.iter import IterDataPipe


@functional_datapipe("patch_image")
class PathImageIterDataPipe(IterDataPipe):
    def __init__(self, source_dp: IterDataPipe):
        self.source_dp = source_dp

    def __iter__(self):
        pass
