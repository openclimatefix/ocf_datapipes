"""Convert GSP to Numpy Batch"""
import logging

from torchdata.datapipes import functional_datapipe
from torchdata.datapipes.iter import IterDataPipe

from ocf_datapipes.utils.consts import BatchKey, NumpyBatch
from ocf_datapipes.utils.utils import datetime64_to_float

logger = logging.getLogger(__name__)


@functional_datapipe("convert_gsp_to_numpy_batch")
class ConvertGSPToNumpyBatchIterDataPipe(IterDataPipe):
    """Convert GSP Xarray to NumpyBatch"""

    def __init__(self, source_datapipe: IterDataPipe):
        """
        Convert GSP Xarray to NumpyBatch object

        Args:
            source_datapipe: Datapipe emitting GSP Xarray object
        """
        super().__init__()
        self.source_datapipe = source_datapipe

    def __iter__(self) -> NumpyBatch:
        """Convert from Xarray to NumpyBatch"""
        logger.debug("Converting GSP to numpy to batch")
        for xr_data in self.source_datapipe:

            example: NumpyBatch = {
                BatchKey.gsp: xr_data.values,
                BatchKey.gsp_t0_idx: xr_data.attrs["t0_idx"],
                BatchKey.gsp_id: xr_data.gsp_id.values,
                BatchKey.gsp_capacity_megawatt_power: xr_data.isel(time_utc=0)[
                    "capacity_megawatt_power"
                ].values,
                BatchKey.gsp_time_utc: datetime64_to_float(xr_data["time_utc"].values),
            }

            # Coordinates
            for batch_key, dataset_key in (
                (BatchKey.gsp_y_osgb, "y_osgb"),
                (BatchKey.gsp_x_osgb, "x_osgb"),
            ):
                if dataset_key in xr_data.coords.keys():
                    values = xr_data[dataset_key].values
                    # Expand dims so EncodeSpaceTime works!
                    example[batch_key] = values  # np.expand_dims(values, axis=1)

            yield example
