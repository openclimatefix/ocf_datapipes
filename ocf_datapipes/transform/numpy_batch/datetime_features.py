"""Datapipes to trigonometric date and time to NumpyBatch"""

import numpy as np
from numpy.typing import NDArray
from torch.utils.data import IterDataPipe, functional_datapipe

from ocf_datapipes.batch import BatchKey


def _get_date_time_in_pi(
    dt: NDArray[np.datetime64],
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    day_of_year = (dt - dt.astype("datetime64[Y]")).astype(int)
    minute_of_day = (dt - dt.astype("datetime64[D]")).astype(int)

    # converting into positions on sin-cos circle
    time_in_pi = (2 * np.pi) * (minute_of_day / (24 * 3600))
    date_in_pi = (2 * np.pi) * (day_of_year / (365 * 24 * 3600))

    return date_in_pi, time_in_pi


@functional_datapipe("add_trigonometric_date_time")
class AddTrigonometricDateTimeIterDataPipe(IterDataPipe):
    """Adds the trigonometric encodings of date of year, time of day to the NumpyBatch"""

    def __init__(self, source_datapipe: IterDataPipe, modality_name: str):
        """
        Adds the sine and cosine of time to the NumpyBatch

        Args:
            source_datapipe: Datapipe of NumpyBatch
            modality_name: Modality to add the time for
        """
        self.source_datapipe = source_datapipe
        self.modality_name = modality_name
        assert self.modality_name in [
            "wind",
        ], f"Trigonometric time not implemented for {self.modality_name}"

    def __iter__(self):
        for np_batch in self.source_datapipe:
            time_utc = np_batch[BatchKey.wind_time_utc]

            times: NDArray[np.datetime64] = time_utc.astype("datetime64[s]")

            date_in_pi, time_in_pi = _get_date_time_in_pi(times)

            # Store
            date_sin_batch_key = BatchKey[self.modality_name + "_date_sin"]
            date_cos_batch_key = BatchKey[self.modality_name + "_date_cos"]
            time_sin_batch_key = BatchKey[self.modality_name + "_time_sin"]
            time_cos_batch_key = BatchKey[self.modality_name + "_time_cos"]

            np_batch[date_sin_batch_key] = np.sin(date_in_pi)
            np_batch[date_cos_batch_key] = np.cos(date_in_pi)
            np_batch[time_sin_batch_key] = np.sin(time_in_pi)
            np_batch[time_cos_batch_key] = np.cos(time_in_pi)

            yield np_batch
