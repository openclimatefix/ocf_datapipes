from torchdata.datapipes import functional_datapipe
from torchdata.datapipes.iter import IterDataPipe

from ocf_datapipes.utils.consts import BatchKey, NumpyBatch
from ocf_datapipes.utils.utils import datetime64_to_float


@functional_datapipe("convert_gsp_to_numpy_batch")
class ConvertGSPToNumpyBatchIterDataPipe(IterDataPipe):
    def __init__(self, source_dp: IterDataPipe):
        super().__init__()
        self.source_dp = source_dp

    def __iter__(self) -> NumpyBatch:
        for xr_data in self.source_dp:
            example: NumpyBatch = {
                BatchKey.gsp: xr_data.values,
                BatchKey.gsp_t0_idx: xr_data.attrs["t0_idx"],
                BatchKey.gsp_id: xr_data.gsp_id.values,
                BatchKey.gsp_capacity_mwp: xr_data.isel(time_utc=0)["capacity_mwp"].values,
                BatchKey.gsp_time_utc: datetime64_to_float(xr_data["time_utc"].values),
            }

            # Coordinates
            for batch_key, dataset_key in (
                (BatchKey.gsp_y_osgb, "y_osgb"),
                (BatchKey.gsp_x_osgb, "x_osgb"),
            ):
                values = xr_data[dataset_key].values
                # Expand dims so EncodeSpaceTime works!
                example[batch_key] = values  # np.expand_dims(values, axis=1)

            yield example
