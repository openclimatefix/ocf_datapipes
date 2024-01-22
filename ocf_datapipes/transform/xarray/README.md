# Xarray Transforms

This module contains the datapipes that take in
`xarray` `DataSet` or `DataArray` and return, usually,
another `DataSet` or `DataArray`. The few exceptions are detailed
below.

## DataPipes that return non-Xarray objects

- `metnet_preprocessor` returns a numpy array of the processed Xarray inputs following the description in the MetNet
  2021 paper.
- `find_contiguous_t0_time_periods` returns pandas DataFrames that contains the contiguous time periods
