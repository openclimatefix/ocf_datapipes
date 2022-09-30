"""Convert NWP to NumpyBatch"""
import numpy as np
from torchdata.datapipes import functional_datapipe
from torchdata.datapipes.iter import IterDataPipe

from ocf_datapipes.utils.consts import BatchKey, NumpyBatch
from ocf_datapipes.utils.utils import datetime64_to_float


@functional_datapipe("convert_nwp_to_numpy_batch")
class ConvertNWPToNumpyBatchIterDataPipe(IterDataPipe):
    """Convert NWP Xarray objects to NumpyBatch ones"""

    def __init__(self, source_datapipe: IterDataPipe):
        """
        Convert NWP Xarray objecs to NumpyBatch ones

        Args:
            source_datapipe: Datapipe emitting NWP Xarray objects
        """
        super().__init__()
        self.source_datapipe = source_datapipe

    def __iter__(self) -> NumpyBatch:
        """Convert from Xarray to NumpyBatch"""
        for xr_data in self.source_datapipe:
            example: NumpyBatch = {
                BatchKey.nwp: xr_data.values,
                BatchKey.nwp_t0_idx: xr_data.attrs["t0_idx"],
            }

            target_time = xr_data.target_time_utc.values
            example[BatchKey.nwp_target_time_utc] = datetime64_to_float(target_time)
            example[BatchKey.nwp_channel_names] = xr_data.channel.values
            example[BatchKey.nwp_step] = (xr_data.step.values / np.timedelta64(1, "h")).astype(
                np.int64
            )
            example[BatchKey.nwp_init_time_utc] = datetime64_to_float(xr_data.init_time_utc.values)

            for batch_key, dataset_key in (
                (BatchKey.nwp_y_osgb, "y_osgb"),
                (BatchKey.nwp_x_osgb, "x_osgb"),
            ):
                if dataset_key in xr_data.coords.keys():
                    example[batch_key] = xr_data[dataset_key].values

            yield example
