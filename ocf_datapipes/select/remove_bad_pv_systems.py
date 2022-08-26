import xarray as xr
from torchdata.datapipes import functional_datapipe
from torchdata.datapipes.iter import IterDataPipe


@functional_datapipe("remove_bad_pv_systems")
class RemoveBadPVSystemsIterDataPipe(IterDataPipe):
    def __init__(self, source_dp: IterDataPipe):
        self.source_dp = source_dp

    def __iter__(self):
        for xr_data in self.source_dp:
            pass
