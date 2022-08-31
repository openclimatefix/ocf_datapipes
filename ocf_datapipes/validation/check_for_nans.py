from torchdata.datapipes.iter import IterDataPipe
from torchdata.datapipes import functional_datapipe
import xarray as xr
from typing import Union
import numpy as np

@functional_datapipe("check_nans")
class CheckNaNsIterDataPipe(IterDataPipe):
    """ Checks, and optionally fills, NaNs in Xarray Dataset"""
    def __init__(self, source_datapipe: IterDataPipe, variable_name: str = None, fill_nans: bool = False):
        """
        Checks and optionally fills NaNs in the data

        Args:
            source_datapipe: Datapipe emitting Xarray Datasets
            variable_name: Optional variable name for debugging
            fill_nans: Whether to fill NaNs with 0 or not
        """
        self.source_datapipe = source_datapipe
        self.variable_name = variable_name
        self.fill_nans = fill_nans

    def __iter__(self) -> Union[xr.DataArray, xr.Dataset]:
        """
        Checks for NaNs in data

        Returns:
            Original Xarray Dataset, after validation, and potentially with NaNs filled
        """
        for xr_data in self.source_datapipe:
            if self.fill_nans:
                xr_data = check_nan_and_fill_warning(data=xr_data, variable_name=self.variable_name)
            check_nan_and_inf(data=xr_data, variable_name=self.variable_name)
            yield xr_data

def check_nan_and_inf(data: xr.Dataset, variable_name: str = None):
    """Check that all values are non NaNs and not infinite"""

    if np.isnan(data).any():
        message = f"Some data values are NaNs. "
        if variable_name is not None:
            message += f" ({variable_name})"

        # find out which example has nans in it
        for i in range(data.shape[0]):
            if np.isnan(data[i]).any():
                message += f" Nans in example {i}."
        raise Exception(message)

    if np.isinf(data).any():
        message = f"Some data values are Infinite"
        if variable_name is not None:
            message += f" ({variable_name})"
        raise Exception(message)


def check_nan_and_fill_warning(data: xr.Dataset, variable_name: str = None) -> xr.Dataset:
    """Check that all values are non NaNs and not infinite"""

    if np.isnan(data).any():
        message = f"Some  data values are NaNs"
        if variable_name is not None:
            message += f" ({variable_name})"
        data = data.fillna(0)

    return data
