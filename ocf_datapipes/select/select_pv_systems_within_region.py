import logging

import numpy as np
import pandas as pd
import xarray as xr
from torchdata.datapipes import functional_datapipe
from torchdata.datapipes.iter import IterDataPipe

_log = logging.getLogger(__name__)


@functional_datapipe("select_pv_systems_within_region")
class SelectPVSystemsWithinRegionIterDataPipe(IterDataPipe):
    def __init__(
        self,
        source_dp: IterDataPipe,
        location_dp: IterDataPipe,
        roi_width_km: float,
        roi_height_km: float,
    ):
        self.source_dp = source_dp
        self.location_dp = location_dp
        self.roi_width_km = roi_width_km
        self.roi_height_km = roi_height_km

    def __iter__(self):
        pass
