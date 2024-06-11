"""Convert PV to Numpy"""

import numpy as np
from torch.utils.data import IterDataPipe, functional_datapipe

from ocf_datapipes.batch import NumpyBatch


@functional_datapipe("convert_pv_to_numpy")
class ConvertPVToNumpyIterDataPipe(IterDataPipe):
    """Convert PV Xarray to NumpyBatch"""

    def __init__(
        self,
        source_datapipe: IterDataPipe,
        return_pv_id: bool = False,
        return_ml_id: bool = False,
    ):
        """
        Convert PV Xarray objects to NumpyBatch objects

        Args:
            source_datapipe: Datapipe emitting PV Xarray objects
            return_pv_id: Whether to return PV ID
            return_ml_id: Whether to return ML ID
        """
        super().__init__()
        self.source_datapipe = source_datapipe
        self.return_pv_id = return_pv_id
        self.return_ml_id = return_ml_id

    def __iter__(self) -> NumpyBatch:
        """Iterate and convert PV Xarray to NumpyBatch"""
        for xr_data in self.source_datapipe:
            pv_yield_history = xr_data.values
            returned_values = [pv_yield_history]
            if self.return_ml_id:
                ml_ids = xr_data["ml_id"].values.astype(np.float32)
                returned_values.append(ml_ids)
            if self.return_pv_id:
                pv_id = xr_data["pv_system_id"].values.astype(np.float32)
                returned_values.append(pv_id)
            yield returned_values
