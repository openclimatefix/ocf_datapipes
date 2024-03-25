"""Datapipes to check equalities"""

import logging
from typing import Optional, Union

import numpy as np
import xarray as xr
from torch.utils.data import IterDataPipe, functional_datapipe

logger = logging.getLogger(__name__)


@functional_datapipe("check_value_equal_to_fraction")
class CheckValueEqualToFractionIterDataPipe(IterDataPipe):
    """Check how much of the input is equal to a given fraction"""

    def __init__(
        self,
        source_datapipe: IterDataPipe,
        value: int,
        fraction: float,
        dataset_name: Optional[str] = None,
    ):
        """
        Check how much of the input is equal to a given fraction

        Args:
            source_datapipe: Datapipe emitting Xarray object
            value: Value to check
            fraction: Fraction for threshold
            dataset_name: Optional dataset name if checking a subset
        """
        self.source_datapipe = source_datapipe
        self.value = value
        self.fraction = fraction
        self.dataset_name = dataset_name

    def __iter__(self) -> Union[xr.Dataset, xr.DataArray]:
        """Check equality"""
        for xr_data in self.source_datapipe:
            self.check_fraction_of_dataset_equals_value(
                xr_data[self.dataset_name] if self.dataset_name is not None else xr_data,
                value=self.value,
                fraction=self.fraction,
            )
            yield xr_data

    def check_fraction_of_dataset_equals_value(self, data: xr.Dataset, value: int, fraction: float):
        """Check data is greater than a certain value"""
        # Get elementwise equality and fraction
        data_equal_to_value = np.isclose(data, value)
        fraction_equal_to_value = data_equal_to_value.mean()
        # Check fraction is greater than threshold
        if fraction_equal_to_value > fraction:
            message = (
                f"Fraction of data equal to {value} is greater than {fraction} in "
                f"{self.source_datapipe.__repr__()} "
            )
            if self.dataset_name is not None:
                message += f"dataset {self.dataset_name}. "
            message += f"The fraction is {fraction_equal_to_value}. "
            raise Warning(message)


@functional_datapipe("check_greater_than_or_equal_to")
class CheckGreaterThanOrEqualToIterDataPipe(IterDataPipe):
    """Check greater than or equal to"""

    def __init__(
        self, source_datapipe: IterDataPipe, min_value: int, dataset_name: Optional[str] = None
    ):
        """
        Check greater than or equal to check

        Args:
            source_datapipe: Datapipe emitting Xarray object
            min_value: Minimum value
            dataset_name: Optional dataset name if checking a subset
        """
        self.source_datapipe = source_datapipe
        self.min_value = min_value
        self.dataset_name = dataset_name

    def __iter__(self) -> Union[xr.Dataset, xr.DataArray]:
        """Check equality"""
        for xr_data in self.source_datapipe:
            self.check_dataset_greater_than_or_equal_to(
                xr_data[self.dataset_name] if self.dataset_name is not None else xr_data,
                min_value=self.min_value,
            )
            yield xr_data

    def check_dataset_greater_than_or_equal_to(self, data: xr.Dataset, min_value: int):
        """Check data is greater than a certain value"""
        if (data < min_value).any():
            message = (
                f"Some data values are less than {min_value} in {self.source_datapipe.__repr__()} "
            )
            if self.dataset_name is not None:
                message += f"dataset {self.dataset_name}. "
            message += f"The minimum value is {data.min()}. "
            raise Warning(message)


@functional_datapipe("check_less_than_or_equal_to")
class CheckLessThanOrEqualToIterDataPipe(IterDataPipe):
    """Check less than or equal to equality"""

    def __init__(
        self, source_datapipe: IterDataPipe, max_value: int, dataset_name: Optional[str] = None
    ):
        """
        Check less than or equal to equality

        Args:
            source_datapipe: Datapipe emitting Xarray object
            max_value: Max value
            dataset_name: Optioanl dataset name if checking a subset
        """
        self.source_datapipe = source_datapipe
        self.max_value = max_value
        self.dataset_name = dataset_name

    def __iter__(self) -> Union[xr.Dataset, xr.DataArray]:
        """Check equality"""
        for xr_data in self.source_datapipe:
            self.check_dataset_less_than_or_equal_to(
                xr_data[self.dataset_name] if self.dataset_name is not None else xr_data,
                max_value=self.max_value,
            )
            yield xr_data

    def check_dataset_less_than_or_equal_to(self, data: xr.Dataset, max_value: int):
        """Check data is less than a certain value"""
        if (data > max_value).any():
            message = (
                f"Some data values are more than {max_value} in {self.source_datapipe.__repr__()} "
            )
            if self.dataset_name is not None:
                message += f"dataset {self.dataset_name}. "
            message += f"The maximum value  is {data.max()}. "
            raise Warning(message)


@functional_datapipe("check_not_equal_to")
class CheckNotEqualToIterDataPipe(IterDataPipe):
    """Check not equal to equality"""

    def __init__(
        self,
        source_datapipe: IterDataPipe,
        value: int,
        dataset_name: Optional[str] = None,
        raise_error: bool = False,
    ):
        """
        Checks not equal to equality on the data

        Args:
            source_datapipe: Datapipe emitting Xarray object
            value: Value to check
            dataset_name: Optional dataset name if checkinga subset
            raise_error: Whether to raise an error or not
        """
        self.source_datapipe = source_datapipe
        self.value = value
        self.dataset_name = dataset_name
        self.raise_error = raise_error

    def __iter__(self) -> Union[xr.Dataset, xr.DataArray]:
        """Check not equal equality"""
        for xr_data in self.source_datapipe:
            self.check_dataset_not_equal(
                xr_data[self.dataset_name] if self.dataset_name is not None else xr_data,
                value=self.value,
                raise_error=self.raise_error,
            )
            yield xr_data

    def check_dataset_not_equal(self, data: xr.Dataset, value: int, raise_error: bool = True):
        """Check data is not equal than a certain value"""
        if np.isclose(data, value).any():
            message = f"Some data values are equal to {value} in {self.source_datapipe.__repr__()} "
            if self.dataset_name is not None:
                message += f"dataset {self.dataset_name}. "
            if raise_error:
                logger.error(message)
                raise Exception(message)
            else:
                logger.warning(message)
