"""Convert Satellite to NumpyBatch"""
from torchdata.datapipes import functional_datapipe
from torchdata.datapipes.iter import IterDataPipe

from ocf_datapipes.utils.consts import BatchKey, NumpyBatch
from ocf_datapipes.utils.utils import datetime64_to_float


@functional_datapipe("convert_satellite_to_numpy_batch")
class ConvertSatelliteToNumpyBatchIterDataPipe(IterDataPipe):
    """Converts Xarray Satellite to NumpyBatch object"""

    def __init__(self, source_datapipe: IterDataPipe, is_hrv: bool = False):
        """
        Converts Xarray satellite object to NumpyBatch object

        Args:
            source_datapipe: Datapipe emitting Xarray satellite objects
            is_hrv: Whether this is HRV satellite data or non-HRV data
        """
        super().__init__()
        self.source_datapipe = source_datapipe
        self.is_hrv = is_hrv

    def __iter__(self) -> NumpyBatch:
        """Convert each example to a NumpyBatch object"""
        for xr_data in self.source_datapipe:
            if self.is_hrv:
                example: NumpyBatch = {
                    BatchKey.hrvsatellite_actual: xr_data.values,
                    BatchKey.hrvsatellite_t0_idx: xr_data.attrs["t0_idx"],
                    BatchKey.hrvsatellite_time_utc: datetime64_to_float(xr_data["time_utc"].values),
                }

                for batch_key, dataset_key in (
                    (BatchKey.hrvsatellite_y_osgb, "y_osgb"),
                    (BatchKey.hrvsatellite_x_osgb, "x_osgb"),
                    (BatchKey.hrvsatellite_y_geostationary, "y_geostationary"),
                    (BatchKey.hrvsatellite_x_geostationary, "x_geostationary"),
                ):
                    # HRVSatellite coords are already float32.
                    example[batch_key] = xr_data[dataset_key].values
            else:
                example: NumpyBatch = {
                    BatchKey.satellite_actual: xr_data.values,
                    BatchKey.satellite_t0_idx: xr_data.attrs["t0_idx"],
                    BatchKey.satellite_time_utc: datetime64_to_float(xr_data["time_utc"].values),
                }

                for batch_key, dataset_key in (
                    (BatchKey.satellite_y_osgb, "y_osgb"),
                    (BatchKey.satellite_x_osgb, "x_osgb"),
                    (BatchKey.satellite_y_geostationary, "y_geostationary"),
                    (BatchKey.satellite_x_geostationary, "x_geostationary"),
                ):
                    # HRVSatellite coords are already float32.
                    example[batch_key] = xr_data[dataset_key].values

            yield example
