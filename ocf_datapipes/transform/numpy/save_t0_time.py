from torchdata.datapipes import functional_datapipe
from torchdata.datapipes.iter import IterDataPipe

from ocf_datapipes.utils.consts import BatchKey, NumpyBatch


@functional_datapipe("save_t0_time")
class SaveT0TimeIterDataPipe(IterDataPipe):
    def __init__(self, source_dp: IterDataPipe):
        self.source_dp = source_dp

    def __iter__(self) -> NumpyBatch:
        for np_batch in self.source_dp:
            pv_t0_idx = np_batch[BatchKey.pv_t0_idx]
            gsp_t0_idx = np_batch[BatchKey.gsp_t0_idx]
            hrvsatellite_t0_idx = np_batch[BatchKey.hrvsatellite_t0_idx]

            np_batch[BatchKey.pv_time_utc_fourier_t0] = np_batch[BatchKey.pv_time_utc_fourier][
                :, pv_t0_idx
            ]
            np_batch[BatchKey.gsp_time_utc_fourier_t0] = np_batch[BatchKey.gsp_time_utc_fourier][
                :, gsp_t0_idx
            ]
            np_batch[BatchKey.hrvsatellite_time_utc_fourier_t0] = np_batch[
                BatchKey.hrvsatellite_time_utc_fourier
            ][:, hrvsatellite_t0_idx]
            yield np_batch
