from torchdata.datapipes.iter import IterDataPipe
from torchdata.datapipes import functional_datapipe
import datetime

@functional_datapipe("load_gsp")
class LoadGSPIterDataPipe(IterDataPipe):
    def __init__(self, gsp_pv_power_zarr_path: str, gsp_id_to_region_id_filename: str,
                 sheffield_solar_region_path: str, threshold_mw: int = 0,
                 sample_period_duration: datetime.timedelta = datetime.timedelta(minutes=30)):
        self.gsp_pv_power_zarr_path = gsp_pv_power_zarr_path
        self.gsp_id_to_region_id_filename = gsp_id_to_region_id_filename
        self.sheffield_solar_region_path = sheffield_solar_region_path
        self.threshold_mw = threshold_mw
        self.sample_period_duration = sample_period_duration

    def __iter__(self):