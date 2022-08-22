import numpy as np
import xarray as xr
from torchdata.datapipes.iter import IterDataPipe


class ReduceNumPVSystemsIterDataPipe(IterDataPipe):
    def __init__(self, source_dp: IterDataPipe, requested_num_pv_systems: int):
        super().__init__()
        self.source_dp = source_dp
        self.requested_num_pv_systems = requested_num_pv_systems
        self.rng = np.random.default_rng()

    def __iter__(self) -> xr.DataArray:
        yield None
