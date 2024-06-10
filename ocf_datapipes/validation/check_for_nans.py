"""Datapipe to check and optionally fill NaNs in data"""

from typing import Union

import numpy as np
import xarray as xr
from torch.utils.data import IterDataPipe, functional_datapipe


@functional_datapipe("check_nans")
class CheckNaNsIterDataPipe(IterDataPipe):
    """Checks, and optionally fills, NaNs in Xarray Dataset"""

    def __init__(
        self,
        source_datapipe: IterDataPipe,
        dataset_name: str = None,
        fill_nans: bool = False,
        fill_value: float = 0.0,
    ):
        """
        Checks and optionally fills NaNs in the data

        Args:
            source_datapipe: Datapipe emitting Xarray Datasets
            dataset_name: Optional name for dataset to check, if None, checks whole dataset
            fill_nans: Whether to fill NaNs with 0 or not
            fill_value: Value to fill NaNs with
        """
        self.source_datapipe = source_datapipe
        self.dataset_name = dataset_name
        self.fill_nans = fill_nans
        self.source_datapipe_name = source_datapipe.__repr__()
        self.fill_value = fill_value

    def __iter__(self) -> Union[xr.DataArray, xr.Dataset]:
        """
        Checks for NaNs in data

        Returns:
            Original Xarray Dataset, after validation, and potentially with NaNs filled
        """
        for xr_data in self.source_datapipe:
            if self.fill_nans:
                if self.dataset_name is None:
                    xr_data = self.fill_nan(data=xr_data, fill_value=self.fill_value)
                else:
                    xr_data[self.dataset_name] = self.fill_nan(
                        data=xr_data[self.dataset_name],
                        fill_value=self.fill_value,
                    )
            self.check_nan_and_inf(
                data=xr_data if self.dataset_name is None else xr_data[self.dataset_name],
            )
            yield xr_data

    def check_nan_and_inf(self, data: xr.Dataset) -> None:
        """Check that all values are non NaNs and not infinite"""

        if np.isnan(data).any():
            if self.dataset_name is None:
                message = f"Some data values are NaNs in datapipe {self.datapipe_name}. "
            else:
                message = (
                    f"Some data values are NaNs in datapipe {self.datapipe_name},"
                    f" dataset {self.dataset_name}. "
                )

            # find out which example has nans in it
            for i in range(data.shape[0]):
                if np.isnan(data[i]).any():
                    message += f" Nans in example {i}."
            raise Warning(message)

        if np.isinf(data).any():
            message = f"Some data values are Infinite in datapipe {self.datapipe_name}."
            raise Warning(message)

    def fill_nan(self, data: xr.Dataset, fill_value: float = 0.0) -> xr.Dataset:
        """Check that all values are non NaNs and not infinite"""

        if np.isnan(data).any():
            data = data.fillna(fill_value)

        return data
