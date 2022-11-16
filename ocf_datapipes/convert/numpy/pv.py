"""Convert PV to Numpy"""
import numpy as np
from torchdata.datapipes import functional_datapipe
from torchdata.datapipes.iter import IterDataPipe

from ocf_datapipes.utils.consts import NumpyBatch


@functional_datapipe("convert_pv_to_numpy")
class ConvertPVToNumpyIterDataPipe(IterDataPipe):
    """Convert PV Xarray to NumpyBatch"""

    def __init__(
        self,
        source_datapipe: IterDataPipe,
        return_pv_id: bool = False,
        return_pv_system_row: bool = False,
    ):
        """
        Convert PV Xarray objects to NumpyBatch objects

        Args:
            source_datapipe: Datapipe emitting PV Xarray objects
            return_pv_id: Whether to return PV Id or now
            return_pv_system_row: Whether to return PV System Row number
        """
        super().__init__()
        self.source_datapipe = source_datapipe
        self.return_pv_id = return_pv_id
        self.return_pv_system_row = return_pv_system_row

    def __iter__(self) -> NumpyBatch:
        """Iterate and convert PV Xarray to NumpyBatch"""
        for xr_data in self.source_datapipe:
            pv_yield_history = xr_data.values
            returned_values = [pv_yield_history]
            if self.return_pv_system_row:
                pv_system_ids = xr_data["pv_system_row_number"].values
                returned_values.append(pv_system_ids)
            if self.return_pv_id:
                pv_id = xr_data["pv_system_id"].values.astype(np.float32)
                returned_values.append(pv_id)
            yield returned_values
