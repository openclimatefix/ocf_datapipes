# Xarray Transforms

This module contains the datapipes that take in
`xarray` `DataSet` or `DataArray` and return, usually,
another `DataSet` or `DataArray`. The few exceptions are detailed
below.

## DataPipes that return non-Xarray objects

- `find_contiguous_t0_time_periods` returns pandas DataFrames that contains the contiguous time periods
