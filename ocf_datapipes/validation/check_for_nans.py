"""Datapipe to check and optionally fill NaNs in data"""
from typing import Union

import numpy as np
import xarray as xr
from torchdata.datapipes import functional_datapipe
from torchdata.datapipes.iter import IterDataPipe


@functional_datapipe("check_nans")
class CheckNaNsIterDataPipe(IterDataPipe):
    """Checks, and optionally fills, NaNs in Xarray Dataset"""

    def __init__(
        self, source_datapipe: IterDataPipe, dataset_name: str = None, fill_nans: bool = False
    ):
        """
        Checks and optionally fills NaNs in the data

        Args:
            source_datapipe: Datapipe emitting Xarray Datasets
            dataset_name: Optional name for dataset to check, if None, checks whole dataset
            fill_nans: Whether to fill NaNs with 0 or not
        """
        self.source_datapipe = source_datapipe
        self.dataset_name = dataset_name
        self.fill_nans = fill_nans

    def __iter__(self) -> Union[xr.DataArray, xr.Dataset]:
        """
        Checks for NaNs in data

        Returns:
            Original Xarray Dataset, after validation, and potentially with NaNs filled
        """
        for xr_data in self.source_datapipe:
            if self.fill_nans:
                if self.dataset_name is None:
                    xr_data = check_nan_and_fill_warning(data=xr_data)
                else:
                    xr_data[self.dataset_name] = check_nan_and_fill_warning(
                        data=xr_data[self.dataset_name]
                    )
            check_nan_and_inf(
                data=xr_data if self.dataset_name is None else xr_data[self.dataset_name]
            )
            yield xr_data


def check_nan_and_inf(data: xr.Dataset):
    """Check that all values are non NaNs and not infinite"""

    if np.isnan(data).any():
        message = "Some data values are NaNs. "

        # find out which example has nans in it
        for i in range(data.shape[0]):
            if np.isnan(data[i]).any():
                message += f" Nans in example {i}."
        raise Exception(message)

    if np.isinf(data).any():
        message = "Some data values are Infinite"
        raise Exception(message)


def check_nan_and_fill_warning(data: xr.Dataset) -> xr.Dataset:
    """Check that all values are non NaNs and not infinite"""

    if np.isnan(data).any():
        data = data.fillna(0)

    return data
