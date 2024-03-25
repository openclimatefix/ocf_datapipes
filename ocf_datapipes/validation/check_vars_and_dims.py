"""Datapipe to check variable and dimension names exist"""

from typing import Iterable, Optional, Union

import xarray as xr
from torch.utils.data import IterDataPipe, functional_datapipe


@functional_datapipe("check_vars_and_dims")
class CheckVarsAndDimsIterDataPipe(IterDataPipe):
    """Check that the variables and dims exist"""

    def __init__(
        self,
        source_datapipe: IterDataPipe,
        expected_dimensions: Optional[Iterable[str]] = None,
        expected_data_vars: Optional[Iterable[str]] = None,
        dataset_name: Optional[str] = None,
    ):
        """
        Checks data vars and dimensions for validation

        Args:
            source_datapipe: Source datapipe emitting a xarray dataset/dataarray
            expected_dimensions: The expected dimensions in the data
            expected_data_vars: The expected data variable names
            dataset_name: Name of subdataset to check, None for checking all of them
        """
        self.source_datapipe = source_datapipe
        self.expected_dimensions = expected_dimensions
        self.expected_data_vars = expected_data_vars
        self.dataset_name = dataset_name

    def __iter__(self) -> Union[xr.DataArray, xr.Dataset]:
        """
        Validate the data and return the validated xarray dataset

        Returns:
            Xarray dataset

        """
        for xr_data in self.source_datapipe:
            if self.dataset_name is None:
                if self.expected_data_vars is not None:
                    xr_data = validate_data_vars(xr_data, self.expected_data_vars)
                if self.expected_dimensions is not None:
                    xr_data = validate_dims(xr_data, self.expected_dimensions)
                    xr_data = validate_coords(xr_data, self.expected_dimensions)
            else:
                if self.expected_data_vars is not None:
                    xr_data[self.dataset_name] = validate_data_vars(
                        xr_data[self.dataset_name], self.expected_data_vars
                    )
                if self.expected_dimensions is not None:
                    xr_data[self.dataset_name] = validate_dims(
                        xr_data[self.dataset_name], self.expected_dimensions
                    )
                    xr_data[self.dataset_name] = validate_coords(
                        xr_data[self.dataset_name], self.expected_dimensions
                    )
            yield xr_data


def validate_data_vars(
    xr_data: Union[xr.DataArray, xr.Dataset], expected_data_vars: Iterable[str]
) -> Union[xr.DataArray, xr.Dataset]:
    """
    Validate the data variables in the Xarray dataset

    Args:
        xr_data: Xarray dataset
        expected_data_vars: The expected data variable names

    Returns:
        Original Xarray dataset
    """
    data_var_names = xr_data.data_vars
    for data_var in expected_data_vars:
        assert data_var in data_var_names, f"{data_var} is not in all data_vars ({data_var_names})!"
    return xr_data


def validate_coords(
    xr_data: Union[xr.DataArray, xr.Dataset], expected_dimensions: Iterable[str]
) -> Union[xr.DataArray, xr.Dataset]:
    """
    Validates that the coordinates exist and are not 0-length in the dataset

    Args:
        xr_data: Xarray dataset
        expected_dimensions: The expected dimensions

    Returns:
        Original Xarray dataset
    """
    for dim in expected_dimensions:
        coord = xr_data.coords[f"{dim}"]
        assert len(coord) > 0, f"{dim} is empty!"
    return xr_data


def validate_dims(
    xr_data: Union[xr.DataArray, xr.Dataset], expected_dimensions: Iterable[str]
) -> Union[xr.DataArray, xr.Dataset]:
    """
    Validate the dimensions exist

    Args:
        xr_data: Xarray dataset to check
        expected_dimensions: The expected dimension names

    Returns:
        The original Xarray Dataset
    """
    assert all(dim in expected_dimensions for dim in xr_data.dims if dim != "example"), (
        f"dims is wrong! "
        f"dims is {xr_data.dims}. "
        f"But we expected {expected_dimensions}."
        " Note that 'example' is ignored, and the order is ignored"
    )
    return xr_data
