"""Save out t0 time"""
from torchdata.datapipes import functional_datapipe
from torchdata.datapipes.iter import IterDataPipe

from ocf_datapipes.utils.consts import BatchKey, NumpyBatch


@functional_datapipe("save_t0_time")
class SaveT0TimeIterDataPipe(IterDataPipe):
    """Save out t0 time"""

    def __init__(self, source_datapipe: IterDataPipe):
        """
        Save out t0 time

        Args:
            source_datapipe: Datapipe of NumpyBatch objectsK
        """
        self.source_datapipe = source_datapipe

    def __iter__(self) -> NumpyBatch:
        for np_batch in self.source_datapipe:
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
